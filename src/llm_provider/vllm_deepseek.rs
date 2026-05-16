use async_stream::try_stream;
use async_trait::async_trait;
use futures_core::Stream;
use futures_util::StreamExt;
use serde_json::Value;
use std::collections::HashMap;
use std::pin::Pin;

use crate::llm_provider::openai_compatible::{client, mapper};
use crate::llm_provider::{Provider, ProviderError, UnifiedEvent, UnifiedResponse};
use crate::openai_http_mapping::validate_openai_request;
use crate::openai_types::{ChatCompletionRequest, ClientError, Content, ContentPart, Role};
use crate::serve_config::ConfigError;

#[derive(Clone)]
pub struct VllmDeepseekProvider {
    client: client::Client,
    model_id: Option<String>,
}

impl VllmDeepseekProvider {
    pub fn new(client: client::Client, model_id: Option<String>) -> Self {
        Self { client, model_id }
    }
}

#[async_trait]
impl Provider for VllmDeepseekProvider {
    fn model_id(&self) -> &str {
        "deepseek-v4-flash"
    }

    async fn complete(
        &self,
        mut request: ChatCompletionRequest,
    ) -> Result<UnifiedResponse, ProviderError> {
        if let Some(model_id) = &self.model_id {
            request.model = model_id.clone();
        }
        let request = normalize_request(request)?;
        validate_request(&request)?;
        let response = self
            .client
            .chat_completions(request)
            .await
            .map_err(map_openai_error)?;
        mapper::map_response(response)
    }

    async fn stream<'a>(
        &'a self,
        mut request: ChatCompletionRequest,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<UnifiedEvent, ProviderError>> + Send + 'a>>,
        ProviderError,
    > {
        if let Some(model_id) = &self.model_id {
            request.model = model_id.clone();
        }
        let request = normalize_request(request)?;
        validate_request(&request)?;
        let stream = self
            .client
            .chat_completions_stream(request)
            .await
            .map_err(map_openai_error)?;

        let output = try_stream! {
            futures_util::pin_mut!(stream);
            let mut state = mapper::StreamMapState::default();

            while let Some(item) = stream.next().await {
                let chunk = item.map_err(map_openai_error)?;
                for event in mapper::map_stream_chunk(chunk, &mut state) {
                    yield event;
                }
            }
        };

        Ok(Box::pin(output))
    }
}

pub fn build_vllm_deepseek_provider(
    params: &std::collections::HashMap<String, String>,
) -> Result<VllmDeepseekProvider, ConfigError> {
    let api_key = params
        .get("api_key")
        .ok_or_else(|| ConfigError::InvalidProvider("api_key is required".to_string()))?;
    let mut config = client::Config::new(api_key.to_string());
    let model_id = params.get("model").cloned();
    if let Some(base_url) = params.get("base_url") {
        config = config.base_url(base_url.to_string());
    }
    let client = client::Client::new(config)
        .map_err(|err| ConfigError::InvalidProvider(err.to_string()))?;
    Ok(VllmDeepseekProvider::new(client, model_id))
}

fn validate_request(request: &ChatCompletionRequest) -> Result<(), ProviderError> {
    validate_openai_request(request).map_err(ProviderError::Internal)
}

fn normalize_request(
    mut request: ChatCompletionRequest,
) -> Result<ChatCompletionRequest, ProviderError> {
    ensure_no_image_parts(&request)?;
    flatten_assistant_text_content(&mut request);
    normalize_reasoning_effort(&mut request);
    Ok(request)
}

fn ensure_no_image_parts(request: &ChatCompletionRequest) -> Result<(), ProviderError> {
    for message in &request.messages {
        if let Content::Parts(parts) = &message.content {
            for part in parts {
                if matches!(part, ContentPart::Image { .. }) {
                    return Err(ProviderError::Internal(
                        "deepseek-v4-flash does not support image_url messages".to_string(),
                    ));
                }
            }
        }
    }
    Ok(())
}

fn flatten_assistant_text_content(request: &mut ChatCompletionRequest) {
    for message in &mut request.messages {
        if message.role != Role::Assistant {
            continue;
        }
        let mut merged_text: Option<String> = None;
        if let Content::Parts(parts) = &message.content {
            let mut combined = String::new();
            let mut all_text = true;
            for part in parts {
                if let ContentPart::Text { text } = part {
                    combined.push_str(text);
                } else {
                    all_text = false;
                    break;
                }
            }
            if all_text {
                merged_text = Some(combined);
            }
        }
        if let Some(text) = merged_text {
            message.content = Content::Text(text);
        }
    }
}

fn normalize_reasoning_effort(request: &mut ChatCompletionRequest) {
    let effort = request.reasoning_effort.take();
    let Some(effort) = effort else {
        return;
    };

    let has_thinking = request
        .chat_template_kwargs
        .as_ref()
        .map(|kwargs| kwargs.contains_key("thinking"))
        .unwrap_or(false);
    if has_thinking {
        return;
    }

    if matches!(effort.as_str(), "low" | "medium" | "high" | "max") {
        let mut kwargs = HashMap::new();
        kwargs.insert("thinking".to_string(), Value::Bool(true));
        kwargs.insert("reasoning_effort".to_string(), Value::String(effort));
        request.chat_template_kwargs = Some(kwargs);
    }
}

fn map_openai_error(err: ClientError) -> ProviderError {
    ProviderError::Internal(err.to_string())
}
