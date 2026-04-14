use anyhow::Result;
use rclaude_context::config::Config;
use rclaude_context::mcp::create_shared_state;
use rclaude_context::mcp::CodebaseTools;
use rmcp::{handler::server::router::Router, ServiceExt};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing to stderr (stdout is reserved for MCP transport)
    tracing_subscriber::registry()
        .with(fmt::layer().with_writer(std::io::stderr))
        .with(EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
        .init();

    tracing::info!("Starting rclaude-context MCP server");

    // Create the tool handler and wrap in Router
    let config = Config::from_env();
    let tools = CodebaseTools::new(create_shared_state(config));
    let router = Router::new(tools);

    // Serve MCP over stdio
    let service = router
        .serve((tokio::io::stdin(), tokio::io::stdout()))
        .await?;

    tracing::info!("MCP server initialized, waiting for requests");

    // Wait for service to complete
    match service.waiting().await {
        Ok(reason) => tracing::info!(?reason, "Server stopped"),
        Err(e) => tracing::error!(?e, "Server task failed"),
    }

    Ok(())
}
