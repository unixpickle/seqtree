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
	m := &Model{BaseFeatures: 2, ExtraFeatures: 1337}
	seq := MakeOneHotSequence(seqInts, 2, 2)
	for i := 0; i <= b.N; i++ {
		BuildTree(m, AllTimesteps(seq), 3, 1, []int{0, 1, 2, 28, 29, 30, 56, 57, 58})
	}
}
