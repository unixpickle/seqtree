package seqtree

import (
	"math/rand"
	"testing"
)

func generateTestModel(baseFeatures int) *Model {
	m := &Model{BaseFeatures: baseFeatures}
	for i := 0; i < 4; i++ {
		b := &Builder{
			Heuristic:       GradientHeuristic{Loss: Softmax{}},
			Depth:           3,
			MinSplitSamples: 10,
			Horizons:        []int{0, 1, 2},
		}
		t := b.Build(TimestepSamples(generateTestSequences(m)))
		m.Add(t, 0.1)
	}
	return m
}

func generateTestSequences(m *Model) []Sequence {
	var res []Sequence
	for i := 0; i < 15; i++ {
		seq := make([]int, 20)
		for i := range seq {
			seq[i] = rand.Intn(m.BaseFeatures)
		}
		res = append(res, MakeOneHotSequence(seq, m.BaseFeatures, m.NumFeatures()))
	}
	m.EvaluateAll(res)
	return res
}

func BenchmarkBuildTreeDense(b *testing.B) {
	seqInts := make([]int, 768)
	for i := range seqInts {
		seqInts[i] = rand.Intn(2)
	}
	m := &Model{BaseFeatures: 2}
	seq := MakeOneHotSequence(seqInts, 2, m.NumFeatures())
	builder := Builder{
		Heuristic: GradientHeuristic{Loss: Softmax{}},
		Depth:     3,
		Horizons:  []int{0, 1, 2, 28, 29, 30, 56, 57, 58},
	}
	b.ResetTimer()
	for i := 0; i <= b.N; i++ {
		builder.Build(TimestepSamples([]Sequence{seq}))
	}
}
