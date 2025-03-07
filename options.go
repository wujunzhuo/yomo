package yomo

import (
	"crypto/tls"
	"log/slog"

	"github.com/quic-go/quic-go"
	"github.com/yomorun/yomo/core"
	"github.com/yomorun/yomo/core/frame"
	"github.com/yomorun/yomo/core/router"
)

type (
	// SourceOption is option for the Source.
	SourceOption core.ClientOption

	// SfnOption is option for the SFN.
	SfnOption core.ClientOption
)

// SourceOption Options.
var (
	// WithCredential sets the credential method for the Source.
	WithCredential = func(payload string) SourceOption { return SourceOption(core.WithCredential(payload)) }

	// WithSourceTLSConfig sets tls config for the Source.
	WithSourceTLSConfig = func(tc *tls.Config) SourceOption { return SourceOption(core.WithClientTLSConfig(tc)) }

	// WithSourceQuicConfig sets quic config for the Source.
	WithSourceQuicConfig = func(qc *quic.Config) SourceOption { return SourceOption(core.WithClientQuicConfig(qc)) }

	// WithLogger sets logger for the Source.
	WithLogger = func(l *slog.Logger) SourceOption { return SourceOption(core.WithLogger(l)) }

	// WithSourceReConnect makes source Connect until success, unless authentication fails.
	WithSourceReConnect = func() SourceOption { return SourceOption(core.WithReConnect()) }
)

// Sfn Options.
var (
	// WithSfnCredential sets the credential method for the Sfn.
	WithSfnCredential = func(payload string) SfnOption { return SfnOption(core.WithCredential(payload)) }

	// WithSfnTLSConfig sets tls config for the Sfn.
	WithSfnTLSConfig = func(tc *tls.Config) SfnOption { return SfnOption(core.WithClientTLSConfig(tc)) }

	// WithSfnQuicConfig sets quic config for the Sfn.
	WithSfnQuicConfig = func(qc *quic.Config) SfnOption { return SfnOption(core.WithClientQuicConfig(qc)) }

	// WithSfnLogger sets logger for the Sfn.
	WithSfnLogger = func(l *slog.Logger) SfnOption { return SfnOption(core.WithLogger(l)) }

	// WithSfnReConnect makes sfn Connect until success, unless authentication fails.
	WithSfnReConnect = func() SfnOption { return SfnOption(core.WithReConnect()) }

	// WithSfnAIFunctionDefinition sets AI function definition for the Sfn.
	WithSfnAIFunctionDefinition = func(description string, inputModel any) SfnOption {
		return SfnOption(core.WithAIFunctionDefinition(description, inputModel))
	}

	// WithAIFunctionJsonDefinition sets AI function definition for the Sfn in the form of jsonschema string.
	WithAIFunctionJsonDefinition = func(jsonschema string) SfnOption {
		return SfnOption(core.WithAIFunctionJsonDefinition(jsonschema))
	}

	// DisableOtelTrace determines whether to disable otel trace.
	DisableOtelTrace = func() SfnOption { return SfnOption(core.DisableOtelTrace()) }
)

// ClientOption is option for the upstream Zipper.
type ClientOption = core.ClientOption

type zipperOptions struct {
	serverOption []core.ServerOption
	clientOption []ClientOption
}

// ZipperOption is option for the Zipper.
type ZipperOption func(*zipperOptions)

var (
	// WithAuth sets the zipper authentication method.
	WithAuth = func(name string, args ...string) ZipperOption {
		return func(zo *zipperOptions) {
			zo.serverOption = append(zo.serverOption, core.WithAuth(name, args...))
		}
	}

	// WithZipperTLSConfig sets the TLS configuration for the zipper.
	WithZipperTLSConfig = func(tc *tls.Config) ZipperOption {
		return func(zo *zipperOptions) {
			zo.serverOption = append(zo.serverOption, core.WithServerTLSConfig(tc))
		}
	}

	// WithZipperQuicConfig sets the QUIC configuration for the zipper.
	WithZipperQuicConfig = func(qc *quic.Config) ZipperOption {
		return func(zo *zipperOptions) {
			zo.serverOption = append(zo.serverOption, core.WithServerQuicConfig(qc))
		}
	}

	// WithZipperLogger sets logger for the zipper.
	WithZipperLogger = func(l *slog.Logger) ZipperOption {
		return func(zo *zipperOptions) {
			zo.serverOption = append(zo.serverOption, core.WithServerLogger(l))
		}
	}

	// WithRouter sets router for the zipper.
	WithRouter = func(r router.Router) ZipperOption {
		return func(zo *zipperOptions) {
			zo.serverOption = append(zo.serverOption, core.WithRouter(r))
		}
	}

	// WithConnector sets connector for the zipper.
	WithConnector = func(c core.Connector) ZipperOption {
		return func(zo *zipperOptions) {
			zo.serverOption = append(zo.serverOption, core.WithConnector(c))
		}
	}

	// WithVersionNegotiateFunc sets the version negotiate function for the zipper
	WithVersionNegotiateFunc = func(f core.VersionNegotiateFunc) ZipperOption {
		return func(zo *zipperOptions) {
			zo.serverOption = append(zo.serverOption, core.WithVersionNegotiateFunc(f))
		}
	}

	// WithUpstreamOption provides upstream zipper options for Zipper.
	WithUpstreamOption = func(opts ...ClientOption) ZipperOption {
		return func(o *zipperOptions) {
			o.clientOption = opts
		}
	}

	// WithZipperConnMiddleware sets conn middleware for the zipper.
	WithZipperConnMiddleware = func(mw ...core.ConnMiddleware) ZipperOption {
		return func(o *zipperOptions) {
			o.serverOption = append(o.serverOption, core.WithConnMiddleware(mw...))
		}
	}

	// WithZipperFrameMiddleware sets frame middleware for the zipper.
	WithZipperFrameMiddleware = func(mw ...core.FrameMiddleware) ZipperOption {
		return func(o *zipperOptions) {
			o.serverOption = append(o.serverOption, core.WithFrameMiddleware(mw...))
		}
	}

	// WithFrameListener adds a Listener other than a quic.Listener.
	WithFrameListener = func(l ...frame.Listener) ZipperOption {
		return func(o *zipperOptions) {
			o.serverOption = append(o.serverOption, core.WithFrameListener(l...))
		}
	}
)
