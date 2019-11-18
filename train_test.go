package seqtree

import (
	"math/rand"
	"testing"
)

func BenchmarkBuildTreeDense(b *testing.B) {
	seqInts := make([]int, 768)
	for i := range seqInts {
		seqInts[i] = rand.Intn(2)
	}
	m := &Model{BaseFeatures: 2}
	seq := MakeOneHotSequence(seqInts, 2, m.NumFeatures())
	builder := Builder{Depth: 3, Horizons: []int{0, 1, 2, 28, 29, 30, 56, 57, 58}}
	for i := 0; i <= b.N; i++ {
		builder.Build(AllTimesteps(seq))
	}
}
