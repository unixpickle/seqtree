package main

import (
	"fmt"
	"os"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/seqtree"
)

func main() {
	if len(os.Args) != 2 {
		essentials.Die("Usage: analysis <model.json>")
	}
	path := os.Args[1]

	model := &seqtree.Model{}
	essentials.Must(model.Load(path))

	for i, tree := range model.Trees {
		fmt.Println(i, TreeMaxHeight(tree), TreeMeanHeight(tree))
	}
}

func TreeMaxHeight(t *seqtree.Tree) int {
	if t.Leaf != nil {
		return 0
	}
	return 1 + essentials.MaxInt(TreeMaxHeight(t.Branch.FalseBranch),
		TreeMaxHeight(t.Branch.TrueBranch))
}

func TreeMeanHeight(t *seqtree.Tree) float64 {
	if t.Leaf != nil {
		return 0
	}
	return 1 + (TreeMeanHeight(t.Branch.FalseBranch)+TreeMeanHeight(t.Branch.TrueBranch))/2
}
