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
)

const Batch = 10000

func main() {
	data := mnist.LoadTrainingDataSet()
	testData := mnist.LoadTestingDataSet()

	seqModel := NewSequenceModel()
	seqModel.Load("model.json")
	for i := 0; true; i++ {
		testSeqs := booleanSamples(testData, Batch)
		testLoss := seqModel.MeanLoss(testSeqs)

		trainSeqs := booleanSamples(data, Batch)
		loss, delta := seqModel.AddTree(trainSeqs)

		log.Printf("tree %d: loss=%f delta=%f test=%f", seqModel.NumTrees()-1, loss, -delta,
			testLoss)
		seqModel.Save("model.json")
		GenerateSamples(seqModel)
	}
}

func booleanSamples(ds mnist.DataSet, n int) [][]bool {
	var res [][]bool
	for _, i := range rand.Perm(len(ds.Samples))[:n] {
		sample := ds.Samples[i]
		seq := make([]bool, SequenceLength)
		for i, x := range sample.Intensities {
			seq[i] = x >= 0.5
		}
		res = append(res, seq)
	}
	return res
}

func GenerateSamples(s *SequenceModel) {
	img := image.NewGray(image.Rect(0, 0, ImageSize*4, ImageSize*4))
	for row := 0; row < 4; row++ {
		for col := 0; col < 4; col++ {
			pixels := s.Sample()
			for i := 0; i < ImageSize; i++ {
				for j := 0; j < ImageSize; j++ {
					pixel := 0
					if pixels[i*ImageSize+j] {
						pixel = 255
					}
					img.SetGray(row*ImageSize+j, col*ImageSize+i, color.Gray{Y: uint8(pixel)})
				}
			}
		}
	}
	w, err := os.Create("sample.png")
	essentials.Must(err)
	defer w.Close()
	essentials.Must(png.Encode(w, img))
}
