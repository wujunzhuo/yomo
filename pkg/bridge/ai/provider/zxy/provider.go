package zxy

import (
	"context"

	_ "github.com/joho/godotenv/autoload"
	"github.com/sashabaranov/go-openai"
	"github.com/yomorun/yomo/core/metadata"

	provider "github.com/yomorun/yomo/pkg/bridge/ai/provider"
)

// check if implements ai.Provider
var _ provider.LLMProvider = &Provider{}

type Provider struct {
	client *openai.Client
}

func NewProvider(apiEndpoint string, apiKey string) *Provider {
	c := openai.DefaultConfig(apiKey)
	c.BaseURL = apiEndpoint

	return &Provider{
		client: openai.NewClientWithConfig(c),
	}
}

// Name returns the name of the provider
func (p *Provider) Name() string {
	return "zxy"
}

// GetChatCompletions implements ai.LLMProvider.
func (p *Provider) GetChatCompletions(ctx context.Context, req openai.ChatCompletionRequest, _ metadata.M) (openai.ChatCompletionResponse, error) {
	req.Model = selectModel(req.Model, len(req.Tools) > 0)
	return p.client.CreateChatCompletion(ctx, req)
}

// GetChatCompletionsStream implements ai.LLMProvider.
func (p *Provider) GetChatCompletionsStream(ctx context.Context, req openai.ChatCompletionRequest, _ metadata.M) (provider.ResponseRecver, error) {
	req.Model = selectModel(req.Model, len(req.Tools) > 0)
	return p.client.CreateChatCompletionStream(ctx, req)
}

func selectModel(model string, withTools bool) string {
	if withTools {
		return "qwen2.5"
	}

	if model == "" || model == "auto" {
		return "deepseek-r1"
	}

	return model
}
