package main

import (
	"image"
	"image/color"
	"image/png"
	"log"
	"math/rand"
	"os"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/mnist"

	"github.com/unixpickle/seqtree"
)

const (
	Batch     = 10
	ImageSize = 28
	Depth     = 3
	MaxKL     = 0.001
	MaxStep   = 20.0
)

func main() {
	horizons := []int{}
	for i := -4; i <= 4; i++ {
		for j := 0; j <= 4; j++ {
			if j == 0 && i < 0 {
				continue
			}
			horizons = append(horizons, i+j*ImageSize)
		}
	}
	dataset := mnist.LoadTrainingDataSet()
	model := &seqtree.Model{BaseFeatures: 256}

	for i := 0; true; i++ {
		seqs := SampleSequences(dataset, model, Batch)
		model.EvaluateAll(seqs)

		var loss float32
		for _, seq := range seqs {
			loss += seq.PropagateLoss()
		}

		tree := seqtree.BuildTree(seqtree.AllTimesteps(seqs...), Depth,
			model.NumFeatures(), horizons)

		// Bound KL on a different batch.
		seqs = SampleSequences(dataset, model, Batch)
		model.EvaluateAll(seqs)

		stepSize := seqtree.BoundedStep(seqtree.AllTimesteps(seqs...), tree, MaxKL, MaxStep)
		model.Add(tree, stepSize)

		log.Printf("step %d: loss=%f step_size=%f", i, loss/Batch, stepSize)
		if i%10 == 0 {
			GenerateSequence(model)
		}
	}
}

func SampleSequences(ds mnist.DataSet, m *seqtree.Model, count int) []seqtree.Sequence {
	var res []seqtree.Sequence
	for i := 0; i < count; i++ {
		sample := ds.Samples[rand.Intn(len(ds.Samples))]
		intSeq := make([]int, len(sample.Intensities))
		for i, intensity := range sample.Intensities {
			if intensity > 0.5 {
				intSeq[i] = 1
			}
		}
		seq := seqtree.MakeOneHotSequence(intSeq, 2, m.NumFeatures())
		res = append(res, seq)
	}
	return res
}

func GenerateSequence(m *seqtree.Model) {
	img := image.NewGray(image.Rect(0, 0, ImageSize*4, ImageSize*4))
	for row := 0; row < 4; row++ {
		for col := 0; col < 4; col++ {
			seq := seqtree.Sequence{
				&seqtree.Timestep{
					Output:   make([]float32, 2),
					Features: make([]bool, m.NumFeatures()),
				},
			}
			for i := 0; i < ImageSize; i++ {
				for j := 0; j < ImageSize; j++ {
					m.EvaluateAt(seq, len(seq)-1)
					num := seqtree.SampleSoftmax(seq[len(seq)-1].Output)
					if num == 1 {
						img.SetGray(row*ImageSize+j, col*ImageSize+i, color.Gray{Y: 255})
					}
					ts := &seqtree.Timestep{
						Output:   make([]float32, 2),
						Features: make([]bool, m.NumFeatures()),
					}
					ts.Features[num] = true
					seq = append(seq, ts)
				}
			}
		}
	}
	w, err := os.Create("sample.png")
	essentials.Must(err)
	defer w.Close()
	essentials.Must(png.Encode(w, img))
}
