package main

import (
	"io/ioutil"
	"log"
	"math/rand"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/seqtree"
)

func main() {
	model := &seqtree.Model{BaseFeatures: 256}
	textData, err := ioutil.ReadFile("/usr/share/dict/words")
	essentials.Must(err)

	for {
		seqs := SampleSequences(textData, model, 10, 20)
		loss := 0.0
		for _, seq := range seqs {
			model.Evaluate(seq)
			loss += seq.PropagateLoss()
		}
		tree := seqtree.BuildTree(AllTimesteps(seqs), 10, 3, model.NumFeatures())
		model.Add(tree, 0.1)
		log.Printf("loss=%f", loss)
	}
}

func SampleSequences(t []byte, m *seqtree.Model, count, length int) []*seqtree.Timestep {
	var res []*seqtree.Timestep
	for i := 0; i < count; i++ {
		start := rand.Intn(len(t) - length)
		intSeq := make([]int, length)
		for j := start; j < start+length; j++ {
			intSeq[j-start] = int(t[j])
		}
		seq := seqtree.MakeOneHotSequence(intSeq, 256, m.ExtraFeatures+256)
		res = append(res, seq)
	}
	return res
}

func AllTimesteps(s []*seqtree.Timestep) []*seqtree.Timestep {
	var res []*seqtree.Timestep
	for _, seq := range s {
		seq.Iterate(func(t *seqtree.Timestep) {
			res = append(res, t)
		})
	}
	return res
}
