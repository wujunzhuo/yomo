package template

import (
	"embed"
	"errors"
	"os"
	"strings"
)

//go:embed go
//go:embed node
var fs embed.FS

var (
	ErrUnsupportedSfnType = errors.New("unsupported sfn type")
	ErrorUnsupportedLang  = errors.New("unsupported lang")
	ErrUnsupportedTest    = errors.New("unsupported test")
	ErrUnsupportedFeature = errors.New("unsupported feature")
)

var (
	SupportedSfnTypes = []string{"llm", "normal"}
	SupportedLangs    = []string{"go", "node"}
)

// get template content
func GetContent(command string, sfnType string, lang string, isTest bool) ([]byte, error) {
	name, err := getTemplateFileName(command, sfnType, lang, isTest)
	if err != nil {
		return nil, err
	}
	f, err := fs.Open(name)
	if err != nil {
		if os.IsNotExist(err) {
			if isTest {
				return nil, ErrUnsupportedTest
			}
			return nil, err
		}
		return nil, err
	}
	defer f.Close()
	_, err = f.Stat()
	if err != nil {
		return nil, err
	}

	return fs.ReadFile(name)
}

func getTemplateFileName(command string, sfnType string, lang string, isTest bool) (string, error) {
	if command == "" {
		command = "init"
	}
	sfnType, err := validateSfnType(sfnType)
	if err != nil {
		return "", err
	}
	lang, err = validateLang(lang)
	if err != nil {
		return "", err
	}
	if lang == "node" && sfnType == "normal" {
		return "", errors.New("language node (-l node) only support type llm (-t llm)")
	}

	sb := new(strings.Builder)
	sb.WriteString(lang)
	sb.WriteString("/")
	sb.WriteString(command)
	sb.WriteString("_")
	sb.WriteString(sfnType)
	if isTest {
		sb.WriteString("_test")
	}
	sb.WriteString(".tmpl")

	// validate the path exists
	name := sb.String()

	return name, nil
}

func validateSfnType(sfnType string) (string, error) {
	if sfnType == "" {
		// default sfn type
		return "llm", nil
	}
	for _, t := range SupportedSfnTypes {
		if t == sfnType {
			return sfnType, nil
		}
	}
	return sfnType, ErrUnsupportedSfnType
}

func validateLang(lang string) (string, error) {
	if lang == "" {
		// default lang
		return "go", nil
	}
	for _, l := range SupportedLangs {
		if l == lang {
			return lang, nil
		}
	}
	return lang, ErrorUnsupportedLang
}
