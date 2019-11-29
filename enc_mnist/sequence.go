package main

import (
	"encoding/json"
	"io/ioutil"
	"os"

	"github.com/pkg/errors"
	"github.com/unixpickle/seqtree"
)

type SequenceModel struct {
	Models []*seqtree.Model
}

func NewSequenceModel() *SequenceModel {
	res := &SequenceModel{}
	for i := 0; i < EncodingDim; i++ {
		res.Models = append(res.Models, &seqtree.Model{
			BaseFeatures: i * EncodingOptions,
		})
	}
	return res
}

func (s *SequenceModel) Save(path string) error {
	data, err := json.Marshal(s)
	if err != nil {
		return errors.Wrap(err, "save model")
	}
	if err := ioutil.WriteFile(path, data, 0755); err != nil {
		return errors.Wrap(err, "save model")
	}
	return nil
}

func (s *SequenceModel) Load(path string) error {
	data, err := ioutil.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return errors.Wrap(err, "load model")
	}
	if err := json.Unmarshal(data, s); err != nil {
		return errors.Wrap(err, "load model")
	}
	return nil
}

func (s *SequenceModel) NumTrees() int {
	return len(s.Models[1].Trees)
}

func (s *SequenceModel) Sample() []int {
	var sample []int
	for _, model := range s.Models {
		ts := s.sampleTimestep(model, sample)
		idx := seqtree.Softmax{}.Sample(ts.Output)
		sample = append(sample, idx)
	}
	return sample
}

func (s *SequenceModel) AddTree(intSeqs [][]int) (loss, delta float32) {
	for _, model := range s.Models {
		seqs := make([]seqtree.Sequence, len(intSeqs))
		for i, intSeq := range intSeqs {
			seqs[i] = seqtree.Sequence{s.sampleTimestep(model, intSeq)}
		}
		model.EvaluateAll(seqs)

		for _, seq := range seqs {
			loss += seq.MeanLoss(seqtree.Softmax{})
		}

		builder := seqtree.Builder{
			Heuristic: seqtree.HessianHeuristic{
				Damping: 1.0,
				Loss:    seqtree.Softmax{},
			},
			Depth:           4,
			MinSplitSamples: 100,
			Horizons:        []int{0},
			MaxUnion:        5,
		}
		tree := builder.Build(seqtree.TimestepSamples(seqs))
		seqtree.ScaleOptimalStep(seqtree.TimestepSamples(seqs), tree, seqtree.Softmax{},
			40.0, 10, 30)
		delta += seqtree.AvgLossDelta(seqtree.TimestepSamples(seqs), tree, seqtree.Softmax{},
			1.0)
		model.Add(tree, 1.0)
	}
	loss /= float32(len(intSeqs))
	return
}

func (s *SequenceModel) sampleTimestep(model *seqtree.Model, seq []int) *seqtree.Timestep {
	numPrefix := model.BaseFeatures / EncodingOptions
	ts := &seqtree.Timestep{
		Features: seqtree.NewBitmap(model.NumFeatures()),
		Output:   make([]float32, EncodingOptions),
		Target:   make([]float32, EncodingOptions),
	}
	for i, s := range seq[:numPrefix] {
		ts.Features.Set(i*EncodingOptions+s, true)
	}
	if len(seq) > numPrefix {
		ts.Target[seq[numPrefix]] = 1
	}
	return ts
}
