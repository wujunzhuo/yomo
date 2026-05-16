use std::sync::Arc;

use async_trait::async_trait;
use axum::http::Method;
use futures_util::StreamExt;
use tokio::sync::OnceCell;
use yup_oauth2::authenticator::DefaultAuthenticator;
use yup_oauth2::{ServiceAccountAuthenticator, read_service_account_key};

use crate::model_api_provider::provider::{
    ModelApiProvider, ProviderBody, ProviderRequest, ProviderResponse, filter_request_headers,
    filter_response_headers, parse_stream_flag,
};
use crate::serve_config::{ConfigError, ProviderConfig};

#[derive(Clone)]
pub struct GenerateContentClient {
    client: reqwest::Client,
    model_id: String,
    project_id: String,
    location: String,
    credentials_file: String,
    upstream_model: String,
    authenticator: Arc<OnceCell<DefaultAuthenticator>>,
}

impl GenerateContentClient {
    pub fn new(
        client: reqwest::Client,
        model_id: String,
        project_id: String,
        location: String,
        credentials_file: String,
        upstream_model: String,
    ) -> Self {
        Self {
            client,
            model_id,
            project_id,
            location,
            credentials_file,
            upstream_model,
            authenticator: Arc::new(OnceCell::new()),
        }
    }

    async fn access_token(&self) -> Result<String, anyhow::Error> {
        let authenticator = self
            .authenticator
            .get_or_try_init(|| async {
                let service_account_key = read_service_account_key(&self.credentials_file).await?;
                let authenticator = ServiceAccountAuthenticator::builder(service_account_key)
                    .build()
                    .await?;
                Ok::<DefaultAuthenticator, anyhow::Error>(authenticator)
            })
            .await?;

        let token = authenticator
            .token(&["https://www.googleapis.com/auth/cloud-platform"])
            .await?;

        token
            .token()
            .map(ToString::to_string)
            .ok_or_else(|| anyhow::anyhow!("missing google access token"))
    }
}

#[async_trait]
impl ModelApiProvider for GenerateContentClient {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn execute(&self, req: ProviderRequest) -> Result<ProviderResponse, anyhow::Error> {
        let token = self.access_token().await?;
        let stream = parse_stream_flag(&req.body);
        let url = if stream {
            vertex_stream_generate_content_url(&self.project_id, &self.location, &self.upstream_model)
        } else {
            vertex_generate_content_url(&self.project_id, &self.location, &self.upstream_model)
        };
        let mut headers = filter_request_headers(req.headers);
        headers.insert(
            axum::http::header::AUTHORIZATION,
            format!("Bearer {token}")
                .parse::<axum::http::HeaderValue>()
                .map_err(|err| anyhow::anyhow!(err))?,
        );
        headers.insert(
            axum::http::header::CONTENT_TYPE,
            "application/json"
                .parse()
                .expect("static header value must be valid"),
        );

        let mut builder = self.client.request(Method::POST, url).headers(headers);
        if !req.body.is_empty() {
            builder = builder.body(req.body);
        }
        let response = builder.send().await?;
        let status = response.status();
        let mut resp_headers = filter_response_headers(response.headers());

        if stream {
            resp_headers.remove(axum::http::header::CONTENT_LENGTH);
            let body_stream = response.bytes_stream().map(|chunk| match chunk {
                Ok(bytes) => Ok(bytes),
                Err(err) => Err(std::io::Error::new(std::io::ErrorKind::Other, err)),
            });

            Ok(ProviderResponse {
                status,
                headers: resp_headers,
                body: ProviderBody::Stream(Box::pin(body_stream)),
            })
        } else {
            let bytes = response.bytes().await?;
            Ok(ProviderResponse {
                status,
                headers: resp_headers,
                body: ProviderBody::Full(bytes),
            })
        }
    }
}

fn vertex_generate_content_url(project_id: &str, location: &str, model: &str) -> String {
    if location == "global" {
        return format!(
            "https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model}:generateContent"
        );
    }

    format!(
        "https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model}:generateContent"
    )
}

fn vertex_stream_generate_content_url(project_id: &str, location: &str, model: &str) -> String {
    format!(
        "{}?alt=sse",
        vertex_base_generate_content_url(project_id, location, model, true)
    )
}

fn vertex_base_generate_content_url(
    project_id: &str,
    location: &str,
    model: &str,
    stream: bool,
) -> String {
    let action = if stream {
        "streamGenerateContent"
    } else {
        "generateContent"
    };

    if location == "global" {
        return format!(
            "https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model}:{action}"
        );
    }

    format!(
        "https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model}:{action}"
    )
}

pub fn build_client(provider: &ProviderConfig) -> Result<Arc<dyn ModelApiProvider>, ConfigError> {
    if provider.provider_type != "generate_content" {
        return Err(ConfigError::UnknownProviderType(provider.provider_type.clone()));
    }
    let project_id = provider
        .params
        .get("project_id")
        .cloned()
        .ok_or_else(|| ConfigError::InvalidProvider("project_id is required".to_string()))?;
    let location = provider
        .params
        .get("location")
        .cloned()
        .unwrap_or_else(|| "global".to_string());
    let credentials_file = provider
        .params
        .get("credentials_file")
        .cloned()
        .ok_or_else(|| ConfigError::InvalidProvider("credentials_file is required".to_string()))?;
    let upstream_model = provider
        .params
        .get("model")
        .cloned()
        .ok_or_else(|| ConfigError::InvalidProvider("model is required".to_string()))?;

    Ok(Arc::new(GenerateContentClient::new(
        reqwest::Client::new(),
        provider.model_id.clone(),
        project_id,
        location,
        credentials_file,
        upstream_model,
    )))
}
