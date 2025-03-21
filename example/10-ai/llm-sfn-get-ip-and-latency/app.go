package main

import (
	"fmt"
	"log/slog"
	"net"

	"github.com/yomorun/yomo/serverless"

	"github.com/go-ping/ping"
)

type Parameter struct {
	Domain string `json:"domain" jsonschema:"description=Domain of the website,example=example.com"`
}

func Description() string {
	return `if user asks ip or network latency of a domain, you should return the result of the giving domain. try your best to dissect user expressions to infer the right domain names`
}

func InputSchema() any {
	return &Parameter{}
}

func Handler(ctx serverless.Context) {
	var msg Parameter
	err := ctx.ReadLLMArguments(&msg)
	if err != nil {
		slog.Error("[sfn] unmarshal arguments", "err", err)
		return
	}

	if msg.Domain == "" {
		slog.Warn("[sfn] domain is empty")
		return
	}

	slog.Info("*fired*", "domain", msg.Domain)

	// get ip of the domain
	ips, err := net.LookupIP(msg.Domain)
	if err != nil {
		slog.Error("[sfn] could not get IPs", "err", err)
		return
	}

	for _, ip := range ips {
		slog.Info("[sfn] get ip", "domain", msg.Domain, "ip", ip)
	}

	// get ip[0] ping latency
	pinger, err := ping.NewPinger(ips[0].String())
	if err != nil {
		slog.Error("[sfn] could not create pinger", "err", err)
		return
	}

	pinger.Count = 3
	pinger.Run()                 // blocks until finished
	stats := pinger.Statistics() // get send/receive/rtt stats

	slog.Info("[sfn] get ping latency", "domain", msg.Domain, "ip", ips[0], "latency", stats.AvgRtt)

	val := fmt.Sprintf("domain %s has ip %s with average latency %s", msg.Domain, ips[0], stats.AvgRtt)

	ctx.WriteLLMResult(val)
}
