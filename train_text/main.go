package main

import (
	"io/ioutil"
	"log"
	"math/rand"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/seqtree"
)

const (
	Batch          = 200
	Length         = 20
	Depth          = 3
	Step           = 0.5
	MinLeafSamples = 2

	WarmupStep  = 5.0
	WarmupSteps = 100
)

var Horizons = []int{0, 1, 2, 3}

func main() {
	model := &seqtree.Model{BaseFeatures: 256}
	textData, err := ioutil.ReadFile("/usr/share/dict/words")
	essentials.Must(err)

	for i := 0; true; i++ {
		seqs := SampleSequences(textData, model, Batch, Length)
		model.EvaluateAll(seqs)

		var loss float32
		for _, seq := range seqs {
			loss += seq.PropagateLoss()
		}

		tree := seqtree.BuildTree(seqtree.AllTimesteps(seqs...), Depth, MinLeafSamples,
			model.NumFeatures(), Horizons)
		if i < WarmupSteps {
			model.Add(tree, WarmupStep)
		} else {
			model.Add(tree, Step)
		}

		log.Printf("step %d: loss=%f", i, loss/Batch)
		if i%10 == 0 {
			GenerateSequence(model, 20)
		}
	}
}

func SampleSequences(t []byte, m *seqtree.Model, count, length int) []seqtree.Sequence {
	var res []seqtree.Sequence
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

func GenerateSequence(m *seqtree.Model, length int) {
	seq := seqtree.Sequence{
		&seqtree.Timestep{
			Output:   make([]float32, 256),
			Features: make([]bool, m.NumFeatures()),
		},
	}
	res := []byte{}
	for i := 0; i < length; i++ {
		m.EvaluateAt(seq, len(seq)-1)
		num := seqtree.SampleSoftmax(seq[len(seq)-1].Output)
		res = append(res, byte(num))
		ts := &seqtree.Timestep{
			Output:   make([]float32, 256),
			Features: make([]bool, m.NumFeatures()),
		}
		ts.Features[num] = true
		seq = append(seq, ts)
	}
	log.Println("sample:", string(res))
}
