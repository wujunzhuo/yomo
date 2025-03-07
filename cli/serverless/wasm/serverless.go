// Package wasm provides WebAssembly serverless function runtimes.
package wasm

import (
	"log"
	"os"
	"sync"

	"github.com/yomorun/yomo"
	cli "github.com/yomorun/yomo/cli/serverless"
	pkglog "github.com/yomorun/yomo/pkg/log"
	"github.com/yomorun/yomo/serverless"
)

// wasmServerless will run serverless functions from the given compiled WebAssembly files.
type wasmServerless struct {
	runtimeName  string
	runtime      Runtime
	name         string
	zipperAddr   string
	observed     []uint32
	wantedTarget string
	credential   string
	mu           sync.Mutex
}

// Setup sets up the serverless
func (s *wasmServerless) Setup(opts *cli.Options) error {
	return nil
}

// Init initializes the serverless
func (s *wasmServerless) Init(opts *cli.Options) error {
	runtime, err := NewRuntime(opts.Runtime)
	if err != nil {
		return err
	}

	err = runtime.Init(opts.Filename)
	if err != nil {
		return err
	}

	s.runtimeName = opts.Runtime
	s.runtime = runtime
	s.name = opts.Name
	s.zipperAddr = opts.ZipperAddr
	s.observed = runtime.GetObserveDataTags()
	s.wantedTarget = runtime.GetWantedTarget()
	s.credential = opts.Credential

	return nil
}

// Build is an empty implementation
func (s *wasmServerless) Build(clean bool) error {
	wasmRuntime := s.runtimeName
	if wasmRuntime == "" {
		wasmRuntime = "wazero"
	}
	pkglog.InfoStatusEvent(os.Stdout, "WASM runtime: %s", wasmRuntime)
	return nil
}

// Run the wasm serverless function
func (s *wasmServerless) Run(verbose bool) error {
	sfn := yomo.NewStreamFunction(
		s.name,
		s.zipperAddr,
		yomo.WithSfnCredential(s.credential),
	)
	// init
	err := sfn.Init(func() error {
		return s.runtime.RunInit()
	})
	if err != nil {
		return err
	}
	// set observe data tags
	sfn.SetObserveDataTags(s.observed...)

	// set wanted target
	sfn.SetWantedTarget(s.wantedTarget)

	sfn.SetHandler(
		func(ctx serverless.Context) {
			s.mu.Lock()
			defer s.mu.Unlock()
			err := s.runtime.RunHandler(ctx)
			if err != nil {
				pkglog.FailureStatusEvent(os.Stderr, "%v", err)
			}
		},
	)

	sfn.SetErrorHandler(
		func(err error) {
			log.Printf("[wasm] error handler: %T %v\n", err, err)
		},
	)

	err = sfn.Connect()
	if err != nil {
		return err
	}
	defer sfn.Close()
	defer s.runtime.Close()

	sfn.Wait()

	return nil
}

// Executable shows whether the program needs to be built
func (s *wasmServerless) Executable() bool {
	return true
}

func init() {
	cli.Register(&wasmServerless{}, ".wasm")
}
