package yomo

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/yomorun/yomo/core"
	"github.com/yomorun/yomo/core/frame"
	"github.com/yomorun/yomo/pkg/config"
)

// Zipper is the orchestrator of yomo. There are two types of zipper:
// one is Upstream Zipper, which is used to connect to multiple downstream zippers,
// another one is Downstream Zipper (will call it as Zipper directly), which is used
// to connected by `Upstream Zipper`, `Source` and `Stream Function`.
type Zipper interface {
	// Logger returns the logger of zipper.
	Logger() *slog.Logger

	// ListenAndServe start zipper as server.
	ListenAndServe(context.Context, string) error

	// Close will close the zipper.
	Close() error
}

// RunZipper run a zipper from a config file.
func RunZipper(ctx context.Context, configPath string) error {
	conf, err := config.ParseConfigFile(configPath)
	if err != nil {
		return err
	}

	// listening address.
	listenAddr := fmt.Sprintf("%s:%d", conf.Host, conf.Port)

	options := []ZipperOption{}
	if _, ok := conf.Auth["type"]; ok {
		if tokenString, ok := conf.Auth["token"]; ok {
			options = append(options, WithAuth("token", tokenString))
		}
	}

	options = append(options, WithZipperFrameMiddleware(core.RejectReservedTagMiddleware))

	zipper, err := NewZipper(conf.Name, conf.Mesh, options...)
	if err != nil {
		return err
	}
	zipper.Logger().Info("using config file", "file_path", configPath)

	return zipper.ListenAndServe(ctx, listenAddr)
}

// NewZipper returns a zipper.
func NewZipper(name string, meshConfig map[string]config.Mesh, options ...ZipperOption) (Zipper, error) {
	opts := &zipperOptions{}

	for _, o := range options {
		o(opts)
	}

	server := core.NewServer(name, opts.serverOption...)

	// add downstreams to server.
	for meshName, meshConf := range meshConfig {
		if meshName == "" || meshName == name {
			continue
		}
		addr := fmt.Sprintf("%s:%d", meshConf.Host, meshConf.Port)

		clientOptions := []core.ClientOption{
			core.WithCredential(meshConf.Credential),
			core.WithNonBlockWrite(),
			core.WithReConnect(),
			core.WithLogger(server.Logger().With("downstream_name", meshName, "downstream_addr", addr)),
		}
		clientOptions = append(clientOptions, opts.clientOption...)

		downstream := &downstream{
			localName: meshName,
			client:    core.NewClient(name, addr, core.ClientTypeUpstreamZipper, clientOptions...),
		}

		server.Logger().Info("add downstream", "downstream_id", downstream.ID(), "downstream_name", downstream.LocalName(), "downstream_addr", addr)

		server.AddDownstreamServer(downstream)
	}

	// watch signal.
	go waitSignalForShutdownServer(server)

	return server, nil
}

func statsToLogger(server *core.Server) {
	logger := server.Logger()

	logger.Info(
		"stats",
		"zipper_name", server.Name(),
		"connector", server.StatsFunctions(),
		"downstreams", server.Downstreams(),
		"data_frame_received_num", server.StatsCounter(),
	)
}

type downstream struct {
	localName string
	client    *core.Client
}

func (d *downstream) Close() error                      { return d.client.Close() }
func (d *downstream) Connect(ctx context.Context) error { return d.client.Connect(ctx) }
func (d *downstream) ID() string                        { return d.client.ClientID() }
func (d *downstream) LocalName() string                 { return d.localName }
func (d *downstream) RemoteName() string                { return d.client.Name() }
func (d *downstream) WriteFrame(f frame.Frame) error    { return d.client.WriteFrame(f) }
