package main

import (
	"io/ioutil"
	"log"
	"math/rand"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/seqtree"
)

const (
	Batch = 50
	Length = 20
	Horizon = 3
	Step = 0.5
)

func main() {
	model := &seqtree.Model{BaseFeatures: 256}
	textData, err := ioutil.ReadFile("/usr/share/dict/words")
	essentials.Must(err)

	for i := 0; true; i++ {
		seqs := SampleSequences(textData, model, Batch, Length)
		loss := 0.0
		for _, seq := range seqs {
			model.Evaluate(seq)
			loss += seq.PropagateLoss()
		}
		tree := seqtree.BuildTree(AllTimesteps(seqs), Horizon, 3, model.NumFeatures())
		model.Add(tree, Step)
		log.Printf("step %d: loss=%f", i, loss/(Batch*Length))
		if i%10 == 0 {
			GenerateSequence(model, 20)
		}
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

func GenerateSequence(m *seqtree.Model, length int) {
	seq := &seqtree.Timestep{
		Output:   make([]float64, 256),
		Features: make([]bool, m.NumFeatures()),
	}
	res := []byte{}
	for i := 0; i < length; i++ {
		m.Evaluate(seq)
		num := seqtree.SampleSoftmax(seq.Output)
		res = append(res, byte(num))
		seq.Next = &seqtree.Timestep{
			Prev:     seq,
			Output:   make([]float64, 256),
			Features: make([]bool, m.NumFeatures()),
		}
		seq.Next.Features[num] = true
		seq = seq.Next
	}
	log.Println("sample:", string(res))
}
