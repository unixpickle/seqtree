package main

import (
	"io/ioutil"
	"log"
	"math/rand"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/seqtree"
)

const (
	Batch           = 2000
	SplitBatch      = 100
	Length          = 20
	Depth           = 4
	MinSplitSamples = 20
	MaxStep         = 40.0
	MaxUnion        = 5
	CandidateSplits = 20
)

var Horizons = []int{0, 1, 2, 3}

func main() {
	model := &seqtree.Model{BaseFeatures: 256}
	model.Load("model.json")

	textData, err := ioutil.ReadFile("/usr/share/dict/words")
	essentials.Must(err)

	builder := seqtree.Builder{
		Depth:           Depth,
		Horizons:        Horizons,
		MinSplitSamples: MinSplitSamples,
		MaxSplitSamples: SplitBatch * Length,
		MaxUnion:        MaxUnion,
		CandidateSplits: CandidateSplits,
	}

	for i := 0; true; i++ {
		seqs := SampleSequences(textData, model, Batch, Length)
		model.EvaluateAll(seqs)

		var loss float32
		for _, seq := range seqs {
			loss += seq.MeanLoss()
		}

		builder.ExtraFeatures = model.ExtraFeatures
		tree := builder.Build(seqtree.TimestepSamples(seqs))

		seqs = SampleSequences(textData, model, Batch, Length)
		model.EvaluateAll(seqs)
		seqtree.ScaleOptimalStep(seqtree.TimestepSamples(seqs), tree, MaxStep, 10, 30)
		delta := seqtree.AvgLossDelta(seqtree.TimestepSamples(seqs), tree, 1.0)
		model.Add(tree, 1.0)

		log.Printf("step %d: loss=%f loss_delta=%f", i, loss/Batch, -delta)
		if i%10 == 0 {
			GenerateSequence(model, Length)
		}
		model.Save("model.json")
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
			Features: seqtree.NewBitmap(m.NumFeatures()),
		},
	}
	res := []byte{}
	for i := 0; i < length; i++ {
		m.EvaluateAt(seq, len(seq)-1)
		num := seqtree.SampleSoftmax(seq[len(seq)-1].Output)
		res = append(res, byte(num))
		ts := &seqtree.Timestep{
			Output:   make([]float32, 256),
			Features: seqtree.NewBitmap(m.NumFeatures()),
		}
		ts.Features.Set(num, true)
		seq = append(seq, ts)
	}
	log.Println("sample:", string(res))
}
