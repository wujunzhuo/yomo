use std::pin::Pin;

use async_trait::async_trait;
use axum::body::Bytes;
use axum::http::{HeaderMap, Method, StatusCode};
use futures_util::stream;
use futures_core::Stream;
use futures_util::StreamExt;
use reqwest::multipart::{Form, Part};
use serde_json::Value;

pub struct ProxyRequest {
    pub method: Method,
    pub endpoint_path: String,
    pub headers: HeaderMap,
    pub body: Bytes,
    pub is_stream: bool,
    pub content_type: Option<String>,
}

pub enum ProxyBody {
    Full(Bytes),
    Stream(Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>>),
}

pub struct ProxyResponse {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub body: ProxyBody,
}

#[async_trait]
pub trait ModelApiProvider: Send + Sync {
    fn model_id(&self) -> &str;

    async fn proxy(&self, req: ProxyRequest) -> Result<ProxyResponse, anyhow::Error>;
}

#[derive(Clone)]
pub struct ProxyClient {
    client: reqwest::Client,
    base_url: String,
    auth_headers: HeaderMap,
    model_id: String,
    upstream_model: String,
}

impl ProxyClient {
    pub fn new(
        client: reqwest::Client,
        base_url: String,
        auth_headers: HeaderMap,
        model_id: String,
        upstream_model: String,
    ) -> Self {
        Self {
            client,
            base_url,
            auth_headers,
            model_id,
            upstream_model,
        }
    }
}

#[async_trait]
impl ModelApiProvider for ProxyClient {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn proxy(&self, req: ProxyRequest) -> Result<ProxyResponse, anyhow::Error> {
        proxy_request(
            &self.client,
            &self.base_url,
            self.auth_headers.clone(),
            Some(self.upstream_model.as_str()),
            req,
        )
        .await
    }
}

const HOP_HEADERS: [&str; 8] = [
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
];

pub async fn proxy_request(
    client: &reqwest::Client,
    base_url: &str,
    mut auth_headers: HeaderMap,
    model_override: Option<&str>,
    req: ProxyRequest,
) -> Result<ProxyResponse, anyhow::Error> {
    let url = format!("{}{}", base_url.trim_end_matches('/'), req.endpoint_path);
    let mut headers = filter_request_headers(req.headers);
    headers.extend(auth_headers.drain());

    let mut request_body = req.body;
    let mut multipart_form: Option<Form> = None;
    if let Some(model) = model_override {
        if let Some(content_type) = req.content_type.as_deref() {
            if content_type.starts_with("application/json") {
                request_body = rewrite_json_model(&request_body, model)?;
            } else if content_type.starts_with("multipart/form-data") {
                multipart_form = Some(rewrite_multipart_model(content_type, &request_body, model).await?);
                headers.remove(axum::http::header::CONTENT_TYPE);
            }
        }
    }

    let mut builder = client.request(req.method, url).headers(headers);
    if let Some(form) = multipart_form {
        builder = builder.multipart(form);
    } else if !request_body.is_empty() {
        builder = builder.body(request_body);
    }

    let response = builder.send().await.map_err(|err| anyhow::anyhow!(err))?;

    let status = response.status();
    let mut resp_headers = filter_response_headers(response.headers());
    let is_stream = req.is_stream;

    if is_stream {
        resp_headers.remove(axum::http::header::CONTENT_LENGTH);
        let stream = response
            .bytes_stream()
            .map(|chunk| match chunk {
                Ok(bytes) => Ok(bytes),
                Err(err) => Err(std::io::Error::new(std::io::ErrorKind::Other, err)),
            });
        let body: Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>> =
            Box::pin(stream);
        Ok(ProxyResponse {
            status,
            headers: resp_headers,
            body: ProxyBody::Stream(body),
        })
    } else {
        let bytes = response.bytes().await.map_err(|err| anyhow::anyhow!(err))?;
        Ok(ProxyResponse {
            status,
            headers: resp_headers,
            body: ProxyBody::Full(bytes),
        })
    }
}

fn rewrite_json_model(body: &Bytes, model: &str) -> Result<Bytes, anyhow::Error> {
    let mut json: Value = serde_json::from_slice(body)?;
    if !json.is_object() {
        return Ok(body.clone());
    }
    json["model"] = Value::String(model.to_string());
    let rewritten = serde_json::to_vec(&json)?;
    Ok(Bytes::from(rewritten))
}

async fn rewrite_multipart_model(
    content_type: &str,
    body: &Bytes,
    model: &str,
) -> Result<Form, anyhow::Error> {
    let boundary = parse_multipart_boundary(content_type)
        .ok_or_else(|| anyhow::anyhow!("multipart boundary is missing"))?;
    let stream = stream::once(async move { Ok::<Bytes, multer::Error>(body.clone()) });
    let mut multipart = multer::Multipart::new(stream, boundary);
    let mut form = Form::new();

    while let Some(field) = multipart.next_field().await? {
        let name = field.name().unwrap_or("").to_string();
        if name == "model" {
            continue;
        }

        let filename = field.file_name().map(|value| value.to_string());
        let mime = field.content_type().map(|value| value.to_string());
        let bytes = field.bytes().await?;

        let mut part = Part::bytes(bytes.to_vec());
        if let Some(filename) = filename {
            part = part.file_name(filename);
        }
        if let Some(mime) = mime {
            part = part.mime_str(&mime)?;
        }
        form = form.part(name, part);
    }

    Ok(form.text("model", model.to_string()))
}

fn parse_multipart_boundary(content_type: &str) -> Option<String> {
    content_type.split(';').find_map(|part| {
        let part = part.trim();
        part.strip_prefix("boundary=")
            .map(|value| value.trim_matches('"').to_string())
    })
}

fn filter_request_headers(headers: HeaderMap) -> HeaderMap {
    let mut filtered = HeaderMap::new();
    for (key, value) in headers.iter() {
        if key == axum::http::header::HOST {
            continue;
        }
        if key == axum::http::header::CONTENT_LENGTH {
            continue;
        }
        if is_hop_header(key.as_str()) {
            continue;
        }
        filtered.insert(key.clone(), value.clone());
    }
    filtered
}

fn filter_response_headers(headers: &HeaderMap) -> HeaderMap {
    let mut filtered = HeaderMap::new();
    for (key, value) in headers.iter() {
        if is_hop_header(key.as_str()) {
            continue;
        }
        filtered.insert(key.clone(), value.clone());
    }
    filtered
}

fn is_hop_header(header: &str) -> bool {
    HOP_HEADERS
        .iter()
        .any(|item| item.eq_ignore_ascii_case(header))
}
