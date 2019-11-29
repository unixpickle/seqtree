package main

import (
	"image"
	"image/color"
	"image/png"
	"log"
	"math"
	"os"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/mnist"
)

func main() {
	dataset := mnist.LoadTrainingDataSet()
	testDataset := mnist.LoadTestingDataSet()

	encoder := NewEncoder()
	encoder.Model.Load("encoder.json")
	if encoder.NeedsTraining() {
		log.Println("Training encoder...")
		TrainEncoder(encoder, dataset)
		encoder.Model.Save("encoder.json")
	}

	seqModel := NewSequenceModel()
	seqModel.Load("sequence_model.json")
	log.Println("Training sequence model...")
	seqs := encoder.EncodeBatch(dataset, len(dataset.Samples))
	testSeqs := encoder.EncodeBatch(testDataset, len(dataset.Samples))
	for {
		testLoss := seqModel.MeanLoss(testSeqs)
		loss, delta := seqModel.AddTree(seqs)
		log.Printf("tree %d: loss=%f delta=%f test=%f", seqModel.NumTrees()-1, loss, -delta,
			testLoss)
		seqModel.Save("sequence_model.json")
		GenerateSamples(encoder, seqModel)
	}
}

func GenerateSamples(e *Encoder, s *SequenceModel) {
	img := image.NewGray(image.Rect(0, 0, ImageSize*4, ImageSize*4))
	for row := 0; row < 4; row++ {
		for col := 0; col < 4; col++ {
			pixels := e.Decode(s.Sample())
			for i := 0; i < ImageSize; i++ {
				for j := 0; j < ImageSize; j++ {
					pixel := 255 / (1 + math.Exp(float64(pixels[i*ImageSize+j])))
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
