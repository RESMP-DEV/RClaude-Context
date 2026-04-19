use std::{borrow::Cow, future::Future, marker::PhantomData, sync::Arc};

use futures::{SinkExt, StreamExt};
use rmcp::{
    service::{RxJsonRpcMessage, ServiceRole, TxJsonRpcMessage},
    transport::Transport,
};
use thiserror::Error;
use tokio::{
    io::{AsyncRead, AsyncWrite},
    sync::Mutex,
};
use tokio_util::{
    bytes::{Buf, BufMut, BytesMut},
    codec::{Decoder, Encoder, FramedRead, FramedWrite},
};

pub type TransportWriter<Role, W> = FramedWrite<W, StdioCodec<TxJsonRpcMessage<Role>>>;

pub struct StdioTransport<Role: ServiceRole, R: AsyncRead, W: AsyncWrite> {
    read: FramedRead<R, StdioCodec<RxJsonRpcMessage<Role>>>,
    write: Arc<Mutex<Option<TransportWriter<Role, W>>>>,
}

impl<Role: ServiceRole, R, W> StdioTransport<Role, R, W>
where
    R: Send + AsyncRead + Unpin,
    W: Send + AsyncWrite + Unpin + 'static,
{
    pub fn new(read: R, write: W) -> Self {
        Self {
            read: FramedRead::new(read, StdioCodec::default()),
            write: Arc::new(Mutex::new(Some(FramedWrite::new(
                write,
                StdioCodec::default(),
            )))),
        }
    }
}

impl<Role: ServiceRole, R, W> Transport<Role> for StdioTransport<Role, R, W>
where
    R: Send + AsyncRead + Unpin,
    W: Send + AsyncWrite + Unpin + 'static,
{
    type Error = StdioTransportError;

    fn name() -> Cow<'static, str> {
        "rust_sindexer::transport::StdioTransport".into()
    }

    fn send(
        &mut self,
        item: TxJsonRpcMessage<Role>,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send + 'static {
        let lock = self.write.clone();
        async move {
            let mut write = lock.lock().await;
            if let Some(ref mut write) = *write {
                write.send(item).await
            } else {
                Err(StdioTransportError::Io(std::io::Error::new(
                    std::io::ErrorKind::NotConnected,
                    "Transport is closed",
                )))
            }
        }
    }

    fn receive(&mut self) -> impl Future<Output = Option<RxJsonRpcMessage<Role>>> + Send {
        let next = self.read.next();
        async {
            next.await.and_then(|result| {
                result
                    .inspect_err(|err| tracing::error!("Error reading from stream: {err}"))
                    .ok()
            })
        }
    }

    async fn close(&mut self) -> Result<(), Self::Error> {
        let mut write = self.write.lock().await;
        drop(write.take());
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct StdioCodec<T> {
    _marker: PhantomData<fn() -> T>,
}

impl<T> Default for StdioCodec<T> {
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

#[derive(Debug, Error)]
pub enum StdioTransportError {
    #[error("missing Content-Length header")]
    MissingContentLength,
    #[error("invalid Content-Length header")]
    InvalidContentLength,
    #[error("serde error {0}")]
    Serde(#[from] serde_json::Error),
    #[error("io error {0}")]
    Io(#[from] std::io::Error),
}

impl<T: serde::de::DeserializeOwned> Decoder for StdioCodec<T> {
    type Item = T;
    type Error = StdioTransportError;

    fn decode(&mut self, buf: &mut BytesMut) -> Result<Option<Self::Item>, Self::Error> {
        loop {
            trim_leading_blank_lines(buf);
            if buf.is_empty() {
                return Ok(None);
            }

            if starts_with_content_length(buf) {
                return decode_content_length_frame(buf);
            }

            if let Some(newline_idx) = buf.iter().position(|byte| *byte == b'\n') {
                let line = buf.split_to(newline_idx + 1);
                let line = trim_ascii_whitespace(&line[..line.len() - 1]);
                if line.is_empty() {
                    continue;
                }
                return Ok(Some(serde_json::from_slice(line)?));
            }

            return Ok(None);
        }
    }

    fn decode_eof(&mut self, buf: &mut BytesMut) -> Result<Option<Self::Item>, Self::Error> {
        if let Some(frame) = self.decode(buf)? {
            return Ok(Some(frame));
        }

        let line = trim_ascii_whitespace(buf);
        if line.is_empty() {
            return Ok(None);
        }

        let parsed = serde_json::from_slice(line)?;
        buf.clear();
        Ok(Some(parsed))
    }
}

impl<T: serde::Serialize> Encoder<T> for StdioCodec<T> {
    type Error = StdioTransportError;

    fn encode(&mut self, item: T, buf: &mut BytesMut) -> Result<(), Self::Error> {
        let body = serde_json::to_vec(&item)?;
        let header = format!("Content-Length: {}\r\n\r\n", body.len());
        buf.reserve(header.len() + body.len());
        buf.put_slice(header.as_bytes());
        buf.put_slice(&body);
        Ok(())
    }
}

fn trim_leading_blank_lines(buf: &mut BytesMut) {
    while matches!(buf.first(), Some(b'\r' | b'\n')) {
        buf.advance(1);
    }
}

fn trim_ascii_whitespace(mut bytes: &[u8]) -> &[u8] {
    while let Some(first) = bytes.first() {
        if first.is_ascii_whitespace() {
            bytes = &bytes[1..];
        } else {
            break;
        }
    }

    while let Some(last) = bytes.last() {
        if last.is_ascii_whitespace() {
            bytes = &bytes[..bytes.len() - 1];
        } else {
            break;
        }
    }

    bytes
}

fn starts_with_content_length(buf: &[u8]) -> bool {
    let prefix = b"content-length:";
    buf.len() >= prefix.len() && buf[..prefix.len()].eq_ignore_ascii_case(prefix)
}

fn decode_content_length_frame<T: serde::de::DeserializeOwned>(
    buf: &mut BytesMut,
) -> Result<Option<T>, StdioTransportError> {
    let (header_end, separator_len) = match find_header_end(buf) {
        Some(value) => value,
        None => return Ok(None),
    };

    let headers = std::str::from_utf8(&buf[..header_end])
        .map_err(|_| StdioTransportError::InvalidContentLength)?;
    let content_length = headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            if name.trim().eq_ignore_ascii_case("content-length") {
                value.trim().parse::<usize>().ok()
            } else {
                None
            }
        })
        .ok_or(StdioTransportError::MissingContentLength)?;

    let frame_len = header_end + separator_len + content_length;
    if buf.len() < frame_len {
        return Ok(None);
    }

    buf.advance(header_end + separator_len);
    let body = buf.split_to(content_length);
    Ok(Some(serde_json::from_slice(&body)?))
}

fn find_header_end(buf: &[u8]) -> Option<(usize, usize)> {
    if let Some(idx) = buf.windows(4).position(|window| window == b"\r\n\r\n") {
        return Some((idx, 4));
    }
    buf.windows(2)
        .position(|window| window == b"\n\n")
        .map(|idx| (idx, 2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_content_length_frames() {
        let mut codec = StdioCodec::<serde_json::Value>::default();
        let body = br#"{"jsonrpc":"2.0"}"#;
        let frame = format!("Content-Length: {}\r\n\r\n", body.len());
        let mut buf = BytesMut::from(frame.as_bytes());
        buf.extend_from_slice(body);

        let msg = codec.decode(&mut buf).unwrap().unwrap();
        assert_eq!(msg["jsonrpc"], "2.0");
        assert!(buf.is_empty());
    }

    #[test]
    fn decodes_newline_delimited_frames() {
        let mut codec = StdioCodec::<serde_json::Value>::default();
        let mut buf = BytesMut::from(&b"{\"jsonrpc\":\"2.0\"}\n"[..]);

        let msg = codec.decode(&mut buf).unwrap().unwrap();
        assert_eq!(msg["jsonrpc"], "2.0");
        assert!(buf.is_empty());
    }

    #[test]
    fn encodes_content_length_frames() {
        let mut codec = StdioCodec::<serde_json::Value>::default();
        let mut buf = BytesMut::new();

        codec
            .encode(serde_json::json!({ "jsonrpc": "2.0" }), &mut buf)
            .unwrap();

        let encoded = String::from_utf8(buf.to_vec()).unwrap();
        assert!(encoded.starts_with("Content-Length: "));
        assert!(encoded.contains("\r\n\r\n"));
        assert!(encoded.ends_with("{\"jsonrpc\":\"2.0\"}"));
    }
}
