package main

import (
	"github.com/unixpickle/seqtree"
)

type SequenceModel struct {
	Model *seqtree.Model
}

func NewSequenceModel() *SequenceModel {
	return &SequenceModel{Model: &seqtree.Model{BaseFeatures: EncodingDim + EncodingOptions}}
}

func (s *SequenceModel) Sample() []int {
	seq := seqtree.Sequence{
		&seqtree.Timestep{
			Output:   make([]float32, EncodingOptions),
			Features: seqtree.NewBitmap(s.Model.NumFeatures()),
		},
	}
	var sample []int
	for i := 0; i < EncodingDim; i++ {
		ts := seq[len(seq)-1]
		ts.Features.Set(EncodingOptions+i, true)
		s.Model.EvaluateAt(seq, len(seq)-1)
		idx := seqtree.Softmax{}.Sample(ts.Output)
		sample = append(sample, idx)
		ts = &seqtree.Timestep{
			Output:   make([]float32, EncodingOptions),
			Features: seqtree.NewBitmap(s.Model.NumFeatures()),
		}
		ts.Features.Set(idx, true)
		seq = append(seq, ts)
	}
	return sample
}

func (s *SequenceModel) AddTree(intSeqs [][]int) (loss, delta float32) {
	seqs := s.sequences(intSeqs)
	s.Model.EvaluateAll(seqs)
	firstSeqs := seqs[:len(seqs)/2]
	secondSeqs := seqs[len(seqs)/2:]

	for _, seq := range seqs {
		loss += seq.MeanLoss(seqtree.Sigmoid{})
	}
	loss /= float32(len(seqs))

	horizons := make([]int, EncodingDim)
	for i := range horizons {
		horizons[i] = i
	}

	builder := seqtree.Builder{
		Heuristic: seqtree.HessianHeuristic{
			Damping: 0.1,
			Loss:    seqtree.Softmax{},
		},
		Depth:           4,
		Horizons:        horizons,
		MinSplitSamples: len(seqs) / 1000,
		MaxUnion:        5,
	}
	tree := builder.Build(seqtree.TimestepSamples(firstSeqs))
	seqtree.ScaleOptimalStep(seqtree.TimestepSamples(secondSeqs), tree, seqtree.Softmax{},
		40.0, 10, 30)
	delta = seqtree.AvgLossDelta(seqtree.TimestepSamples(secondSeqs), tree, seqtree.Softmax{}, 1.0)
	s.Model.Add(tree, 1.0)

	return
}

func (s *SequenceModel) sequences(seqs [][]int) []seqtree.Sequence {
	res := make([]seqtree.Sequence, len(seqs))
	for i, intSeq := range seqs {
		seq := seqtree.Sequence{}
		prev := -1
		for j, x := range intSeq {
			ts := &seqtree.Timestep{
				Output:   make([]float32, EncodingOptions),
				Features: seqtree.NewBitmap(s.Model.NumFeatures()),
				Target:   make([]float32, EncodingOptions),
			}
			if prev != -1 {
				ts.Features.Set(prev, true)
			}
			ts.Features.Set(EncodingOptions+j, true)
			prev = x
			ts.Target[prev] = 1
			seq = append(seq, ts)
		}
		res[i] = seq
	}
	return res
}
