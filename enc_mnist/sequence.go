package main

import (
	"encoding/json"
	"io/ioutil"
	"math/rand"
	"os"

	"github.com/pkg/errors"
	"github.com/unixpickle/seqtree"
)

type SequenceModel struct {
	Models []*seqtree.Model
}

func NewSequenceModel() *SequenceModel {
	res := &SequenceModel{}
	for i := 0; i < EncodingDim1; i++ {
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
		model.Evaluate(seqtree.Sequence{ts})
		idx := seqtree.Softmax{}.Sample(ts.Output)
		sample = append(sample, idx)
	}
	return sample
}

func (s *SequenceModel) AddTree(intSeqs [][]int) (loss, delta float32) {
	for _, model := range s.Models {
		shrinkage := float32(0.1)
		if len(model.Trees) == 0 {
			// Take a big initial step.
			shrinkage = 1.0
		}

		seqs := make([]seqtree.Sequence, len(intSeqs))
		perm := rand.Perm(len(intSeqs))
		for i, intSeq := range intSeqs {
			seqs[perm[i]] = seqtree.Sequence{s.sampleTimestep(model, intSeq)}
		}
		model.EvaluateAll(seqs)

		trainSeqs := seqs[len(seqs)/2:]
		validSeqs := seqs[:len(seqs)/2]

		for _, seq := range seqs {
			loss += seq.MeanLoss(seqtree.Softmax{})
		}

		builder := seqtree.Builder{
			Heuristic: seqtree.HessianHeuristic{
				Damping: 0.1,
				Loss:    seqtree.Softmax{},
			},
			Depth:           5,
			MinSplitSamples: 100,
			Horizons:        []int{0},
			MaxUnion:        5,
		}
		pruner := seqtree.Pruner{
			Heuristic: builder.Heuristic,
			MaxLeaves: 10,
		}
		tree := builder.Build(seqtree.TimestepSamples(trainSeqs))
		tree = pruner.Prune(seqtree.TimestepSamples(validSeqs), tree)
		seqtree.ScaleOptimalStep(seqtree.TimestepSamples(validSeqs), tree, seqtree.Softmax{},
			40.0, 10, 30)
		delta += seqtree.AvgLossDelta(seqtree.TimestepSamples(validSeqs), tree, seqtree.Softmax{},
			shrinkage)
		model.Add(tree, shrinkage)
	}
	loss /= float32(len(intSeqs))
	return
}

func (s *SequenceModel) MeanLoss(intSeqs [][]int) float32 {
	var loss float32
	for _, model := range s.Models {
		seqs := make([]seqtree.Sequence, len(intSeqs))
		for i, intSeq := range intSeqs {
			seqs[i] = seqtree.Sequence{s.sampleTimestep(model, intSeq)}
		}
		model.EvaluateAll(seqs)

		for _, seq := range seqs {
			loss += seq.MeanLoss(seqtree.Softmax{})
		}
	}
	return loss / float32(len(intSeqs))
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
