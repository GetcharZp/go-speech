package melotts

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// loadLexicon 加载发音词典
//
// 数据格式: word phone1 phone2 ... tone1 tone2 ...
func loadLexicon(path string) (map[string]LexiconItem, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	lex := make(map[string]LexiconItem)
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		parts := strings.Fields(line)
		if len(parts) < 2 {
			continue
		}

		word := parts[0]
		rest := parts[1:]

		// 音素数量必须等于声调数量，所以剩余部分必须是偶数
		if len(rest)%2 != 0 {
			fmt.Printf("Skipping invalid lexicon line: %s\n", word)
			continue
		}

		mid := len(rest) / 2
		phones := rest[:mid]
		toneStrings := rest[mid:]

		var tones []int64
		for _, tStr := range toneStrings {
			tVal, _ := strconv.ParseInt(tStr, 10, 64)
			tones = append(tones, tVal)
		}
		lex[word] = LexiconItem{Phones: phones, Tones: tones}
	}
	return lex, scanner.Err()
}

// loadTokens 加载 Token ID 映射表
//
// 数据格式: token id
func loadTokens(path string) (map[string]int64, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	m := make(map[string]int64)
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		parts := strings.Fields(line)
		if len(parts) >= 2 {
			var id int64
			if _, err := fmt.Sscanf(parts[1], "%d", &id); err == nil {
				m[parts[0]] = id
			}
		}
	}
	return m, scanner.Err()
}
