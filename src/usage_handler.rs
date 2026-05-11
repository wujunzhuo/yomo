use async_trait::async_trait;
use serde_json::Value;

#[async_trait]
pub trait UsageHandler<M>: Send + Sync {
    async fn on_usage(
        &self,
        endpoint: &str,
        model_id: &str,
        label: Option<&str>,
        request_id: &str,
        trace_id: &str,
        status_code: u16,
        metadata: M,
        usage: Value,
    );
}

#[derive(Clone, Default)]
pub struct NoopUsageHandler;

#[async_trait]
impl<M> UsageHandler<M> for NoopUsageHandler
where
    M: Send + Sync + 'static,
{
    async fn on_usage(
        &self,
        _endpoint: &str,
        _model_id: &str,
        _label: Option<&str>,
        _request_id: &str,
        _trace_id: &str,
        _status_code: u16,
        _metadata: M,
        _usage: Value,
    ) {
    }
}
