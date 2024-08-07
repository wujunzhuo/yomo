package register

import (
	"testing"

	"github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/yomorun/yomo/ai"
)

func TestRegister(t *testing.T) {
	r := NewDefault()

	SetRegister(r)
	assert.Equal(t, r, GetRegister())

	functionDefinition := &ai.FunctionDefinition{
		Name:        "function1",
		Description: "desc1",
		Parameters: &ai.FunctionParameters{
			Type: "type1",
			Properties: map[string]*ai.ParameterProperty{
				"prop1": {Type: "type1", Description: "desc1"},
				"prop2": {Type: "type2", Description: "desc2"},
			},
			Required: []string{"prop1"},
		},
	}

	err := RegisterFunction(1, functionDefinition, 1, nil)
	assert.NoError(t, err)

	toolCalls, err := ListToolCalls(nil)
	assert.NoError(t, err)
	assertToolCalls(t, 1, functionDefinition, toolCalls)

	UnregisterFunction(1, nil)
	toolCalls, err = ListToolCalls(nil)
	assert.NoError(t, err)
	assertToolCalls(t, 0, nil, toolCalls)
}

func assertToolCalls(t *testing.T, wantTag uint32, want *ai.FunctionDefinition, toolCalls map[uint32]openai.Tool) {
	var (
		tag uint32
		got openai.Tool
	)
	for k, v := range toolCalls {
		tag = k
		got = v
	}
	assert.Equal(t, wantTag, tag)
	assert.Equal(t, want, got.Function)
}
