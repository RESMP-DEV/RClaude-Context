use anyhow::Result;
use rmcp::transport::io::stdio;
use rmcp::ServiceExt;
use sindexer::config::Config;
use sindexer::mcp::create_shared_state;
use sindexer::mcp::CodebaseTools;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(fmt::layer().with_writer(std::io::stderr))
        .with(EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
        .init();

    tracing::info!("Starting sindexer MCP server");

    let config = Config::from_env();
    let tools = CodebaseTools::with_state(create_shared_state(config));
    let service = tools.serve(stdio()).await?;

    tracing::info!("MCP server initialized, waiting for requests");

    match service.waiting().await {
        Ok(reason) => tracing::info!(?reason, "Server stopped"),
        Err(e) => tracing::error!(?e, "Server task failed"),
    }

    Ok(())
}
