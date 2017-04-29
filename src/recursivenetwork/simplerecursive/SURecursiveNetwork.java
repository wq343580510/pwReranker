package recursivenetwork.simplerecursive;

import edu.stanford.nlp.util.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.TanhDerivative;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.AdaGrad;
import reranker.CFGRerankingTree;
import reranker.SURecursiveNetworkModel;
import reranker.TreeLattice;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;


/**
 * Created by zhouh on 15-12-12.
 * <p>
 * A Recursive Neural Net with multi W Matrix according to the children labels
 */
public class SURecursiveNetwork {
    private final int n;
    public final SURecursiveNetworkModel model;

    /** a value that is saved globally for optimization purpose. This is however a bad
     * programming choice and should be removed in later versions.*/
//    private INDArray nonLinearDerivativeTranspose;

    /**
     * for a binarized CFGRerankingTree, W is in Rnx2n and b is in Rn
     */
    public SURecursiveNetwork(int n, SURecursiveNetworkModel model) {
        this.n = n;
        this.model = model;

    }

    /**
     * we do not use this function in the SURNN
     *
     * @param left
     * @param right
     * @return
     */
    public INDArray preOutput(INDArray left, INDArray right) {
        //perreOutput = this.W.mmul(concat).add(this.b).transpose();
        throw new RuntimeException(" we do not use this function in the SURNN");
    }

    /**
     * compute the vectors of the given CFGRerankingTree
     * Assumes that the children vector of this CFGRerankingTree have been computed
     *
     * @param currentTree
     * @return
     */
    public INDArray preOutput(CFGRerankingTree currentTree) {

        INDArray preOutput;
        int numChildren = currentTree.numChildren();

        /*
         * compute W * (a;b) for a given CFGRerankingTree
         */
        if (numChildren == 0)
            preOutput = model.getWordVector(currentTree);
        else if (numChildren == 1) {
            INDArray bias = Nd4j.ones(1, 1);
            INDArray concat = Nd4j.vstack(currentTree.getChild(0).getVector(), bias);
            preOutput = this.model.getUnaryTransform(currentTree).mmul(concat);
        } else {  // numChild == 2
            //perform composition
            INDArray bias = Nd4j.ones(1, 1);
            INDArray concat = Nd4j.vstack(
                    currentTree.getChild(0).getVector(),
                    currentTree.getChild(1).getVector(),
                    bias);
            preOutput = this.model.getBinaryTransform(currentTree).mmul(concat);

        }
        return preOutput;
    }

    public INDArray pwgPreOutput(CFGRerankingTree currentTree) {

        INDArray preOutput;
        int numChildren = currentTree.numChildren();

        /*
         * compute W * (a;b) for a given CFGRerankingTree
         */
        if (numChildren == 0)
            preOutput = model.getWordVector(currentTree);
        else if (numChildren == 1) {
            INDArray bias = Nd4j.ones(1, 1);
            INDArray concat = Nd4j.vstack(currentTree.getChild(0).getGatedVector(), bias);
            preOutput = this.model.getUnaryTransform(currentTree).mmul(concat);
        } else {  // numChild == 2
            //perform composition
            INDArray bias = Nd4j.ones(1, 1);
            INDArray concat = Nd4j.vstack(
                    currentTree.getChild(0).getGatedVector(),
                    currentTree.getChild(1).getGatedVector(),
                    bias);
            preOutput = this.model.getBinaryTransform(currentTree).mmul(concat);

        }
        return preOutput;
    }

    public INDArray applyNonLinearity(INDArray preOutput) {
//        //Be careful that execAndReturn works on the same copy so duplicate the INDArray
//        INDArray nonLinear = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("tanh", preOutput.dup()));
//        return nonLinear;
        return null;
    }

    /**
     * pair-wise gated forward with two different trees
     *
     * @param t
     * @return
     */
    public double secondStageForward(CFGRerankingTree t, TreeLattice comparedLattice) {

        if (t.numChildren() == 0) { //leaf vector, we do not score the leaf node

            return 0;

        } else if (t.numChildren() == 1) { //unary

            double childrenScoreSum = this.secondStageForward(t.getChild(0), comparedLattice);

            INDArray gatedVector = getGatedVector(t, comparedLattice);
            INDArray nonLinear = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("tanh", gatedVector.dup()));
            t.setGatedVector(nonLinear);

            //perform user defined composition
            double currentNodeScore = model.getUnaryScoreLayer(t).mmul(nonLinear).getDouble(0);

            return childrenScoreSum + currentNodeScore;
        } else { //binary

            //do these recursive calls in parallel in future
            double leftChildrenScoreSum = this.secondStageForward(t.getChild(0), comparedLattice);
            double rightChildrenScoreSum = this.secondStageForward(t.getChild(1), comparedLattice);

            //perform user defined composition
            INDArray gatedVector = getGatedVector(t, comparedLattice);
            INDArray nonLinear = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("tanh", gatedVector.dup()));
            t.setGatedVector(nonLinear);

            double currentNodeScore = model.getBinaryScoreLayer(t).mmul(nonLinear).getDouble(0);

            return currentNodeScore + leftChildrenScoreSum + rightChildrenScoreSum;
        }
    }

    private INDArray getGatedVector(CFGRerankingTree t, TreeLattice comparedLattice) {

        INDArray forgetGateW = model.getForgetGate(t);
        //insertForgetGate forgetGateW 200*401
        INDArray concatVector4Gate = getConcatVector4Gate(t, comparedLattice);
        t.setConcatVector4Gate(concatVector4Gate);

        return forgetGateW.mmul(concatVector4Gate);

    }

    private INDArray getConcatVector4Gate(CFGRerankingTree t, TreeLattice comparedLattice) {
        INDArray gate = Nd4j.zeros(n, 1);

        double spanLen = t.end + 1 - t.begin;

        // get the gate vector by max pooling
        List<CFGRerankingTree> inSpanTrees = comparedLattice.getSpanTrees(t.begin, t.end);
        double[] weights = new double[inSpanTrees.size()];

        for (int i = 0; i < inSpanTrees.size(); i++) {
            CFGRerankingTree tree = inSpanTrees.get(i);
            double weight = (tree.end + 1 - tree.begin) / spanLen;
            weights[i] = weight;

            gate.addi(tree.getVector().mul(weight));

        }

        t.setGateSpamTree(inSpanTrees);
        t.setGateWeights(weights);

        INDArray bias = Nd4j.ones(1, 1);
        INDArray concat = Nd4j.vstack(
                t.getVector(),
                gate,
                bias);

        return concat;
    }

    /**
     * feedforward the CFGRerankingTree neural net and all the trees are included in trees
     * In this function, we do not calculate the score of a reranking tree, which will be
     * calculated in the next pairwise forward stage
     *
     * @param t
     * @return
     */
    public double firstStageForward(CFGRerankingTree t, List<CFGRerankingTree> trees) {

        if (t.numChildren() == 0) { //leaf vector, we do not score the leaf node

            INDArray preOutput = this.preOutput(t);
            t.setPreOutput(preOutput);
            INDArray nonLinear = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("tanh", preOutput.dup()));
            t.setVector(nonLinear);
            trees.add(t);
            return 0;

        } else if (t.numChildren() == 1) { //unary

            double childrenScoreSum = this.firstStageForward(t.getChild(0), trees);

            //perform user defined composition
            INDArray preOutput = this.preOutput(t);
            t.setPreOutput(preOutput);
            INDArray nonLinear = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("tanh", preOutput.dup()));
            t.setVector(nonLinear);
            trees.add(t);

            double currentNodeScore = model.getUnaryScoreLayer2(t).mmul(nonLinear).getDouble(0);

            return childrenScoreSum + currentNodeScore;

        } else { //binary

            //do these recursive calls in parallel in future
            double leftChildrenScoreSum = this.firstStageForward(t.getChild(0), trees);
            double rightChildrenScoreSum = this.firstStageForward(t.getChild(1), trees);

            //perform user defined composition
            INDArray preOutput = this.preOutput(t);
            t.setPreOutput(preOutput);
            INDArray nonLinear = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("tanh", preOutput.dup()));
            t.setVector(nonLinear);

            trees.add(t);

            double currentNodeScore = model.getBinaryScoreLayer2(t).mmul(nonLinear).getDouble(0);

            return currentNodeScore + leftChildrenScoreSum + rightChildrenScoreSum;
        }
    }

    public void pairwiseBackProp(CFGRerankingTree t1, CFGRerankingTree t2, SURecursiveNetworkModel gradient3, SURecursiveNetworkModel gradient4) {

        INDArray errorArray = Nd4j.zeros(n, 1);
        pairwiseBackProp(t1, errorArray, gradient3, 1.0);
        pairwiseBackProp(t2, errorArray, gradient4, 1.0);
        pairwiseBackProp(t1, errorArray, gradient3, 0.0);
    }

    public void pairwiseBackProp(CFGRerankingTree t, INDArray error, SURecursiveNetworkModel gradient, double gradientRate) {

        if (t.numChildren() == 0) {
            //fine tune leaf vector embeddings
            if (t.pairWiseGradient != null) {
                error.addi(t.pairWiseGradient);
                t.clearPairWiseGradient();
            }
            gradient.getOrInsertWordVector(t).addi(error);
            return;
        } else {  // children num  ==  1 or 2

            boolean bBinary = t.numChildren() == 2;
            //tanh（wf * (v_t,v_sum)）
            INDArray currentTreeNodeNonlinearVector = t.getGatedVector();
            //nonLinear Derivative = [g'(Wx+b)]
            //Be careful that execAndReturn works on the same copy so duplicate the INDArray
            //tanh'（wf * (v_t,v_sum)）
            INDArray currentTreeNodeNonlinearVectorDerivative = Nd4j.getExecutioner()
                    .execAndReturn(new TanhDerivative(currentTreeNodeNonlinearVector.dup()));

            /*
             * compute the gradients according to current node score
             */
            INDArray scoreLayer = bBinary ? model.getBinaryScoreLayer(t) : model.getUnaryScoreLayer(t);
            INDArray scoreLayer2 = bBinary ? model.getBinaryScoreLayer2(t) : model.getUnaryScoreLayer2(t);
            INDArray transformLayer = bBinary ? model.getBinaryTransform(t) : model.getUnaryTransform(t);
            INDArray forgetGate = model.getForgetGate(t);

            //S_1 * tanh'（wf * (v_t,v_sum)）* rate
            INDArray currentScoreGradients = scoreLayer.transpose().mul(currentTreeNodeNonlinearVectorDerivative);
            currentScoreGradients.muli(gradientRate);

            /*
             * get the gradients for preout put and pairwise gate vector
             */
            //S_1 * tanh'（wf * (v_t,v_sum)）* rate * (v_t,v_sum)
            // the gradient for forget gate
            INDArray gateGradients = currentScoreGradients.mmul(t.concatVector4Gate.transpose());
            gradient.getOrInsertForgetGate(t).addi(gateGradients);

            // wf * S_1 * tanh'（wf * (v_t,v_sum)）* rate
            INDArray concatGateGradients = forgetGate.transpose().mmul(currentScoreGradients);

            // get tanh derivative for the concatenate layer
            // tanh'(v_t)
            INDArray leftTanhDerivative = Nd4j.getExecutioner()
                    .execAndReturn(new TanhDerivative(t.getVector().dup()));
            //tanh'(v_sum)
            INDArray rightTanhDerivative = Nd4j.getExecutioner()
                    .execAndReturn(new TanhDerivative(rightDerivative(t.concatVector4Gate)));
            // wf * S_1 * tanh'（wf * (v_t)）* rate * tanh'(v_t)
            INDArray preOutputGradient = leftDerivative(concatGateGradients).muli(leftTanhDerivative);
            // wf * S_1 * tanh'（wf * (v_sum)）* rate * tanh'(v_sum)
            INDArray gateVectorGradients = rightDerivative(concatGateGradients).muli(rightTanhDerivative);

            // add the derivative of non-gated score layer
            // wf * S_1 * tanh'（wf * (v_t)）* rate * tanh'(v_t) + S_2 * tanh'(v_t) * rate
            //相当于目标函数对preout_t的导数 v_t = tanh(preout_t)
            preOutputGradient.addi(scoreLayer2.transpose().mul(leftTanhDerivative).muli(gradientRate));

            //wf * S_1 * tanh'（wf * (v_sum)）* rate * tanh'(v_sum) * weight_i
            //目标函数对这个span中的一个vector的梯度
            //span中加权的梯度，why需要保存到一个新的字段中
            for (int i = 0; i < t.gateWeights.length; i++) {
                INDArray oneSpamTreeGradient = gateVectorGradients.mul(t.gateWeights[i]);
                t.gateSpamTree.get(i).setPairWiseGradient(oneSpamTreeGradient);
            }

            /*
             * get the gradients for children
             *//*
             * get the total gradients of current linear layer of nets
             * the total gradients include gradient from parent : error
             *                         and from the score layer : currentScoreGradients
             */
            // error（父亲传来的梯度） + wf * S_1 * tanh'（wf * (v_t)）* rate * tanh'(v_t) + S_2 * tanh'(v_t) * rate
            INDArray totalPreoutputGradents = preOutputGradient.add(error);
            //比如最下面一层节点在计算梯度的时候，上面许多节点都用到了他，所以每次就去取梯度，然后设为空
            // the pairWiseGradient are the gradients from the pair-wise gated tree
            if (t.pairWiseGradient != null) {
                totalPreoutputGradents.addi(t.pairWiseGradient);
                t.clearPairWiseGradient();
            }
            // W_t * [error（父亲传来的梯度） + wf * S_1 * tanh'（wf * (v_t)）* rate * tanh'(v_t) + S_2 * tanh'(v_t) * rate]
            INDArray childrenGradients = transformLayer.transpose().mmul(preOutputGradient);
            //这个式子应该是目标函数对 children的表示的梯度，也就是对[v_left,v_right]组合的梯度

            /*
             * update the gradients of current CFGRerankingTree node
             */
            if (t.numChildren() == 1) {
                //tanh（wf * (v_t,v_sum)）* rate  //the gradient of s1
                gradient.getOrInsertUnaryScoreLayer(t).addi(currentTreeNodeNonlinearVector.transpose().muli(gradientRate));
                //v_t * rate //the gradient of s2
                gradient.getOrInsertUnaryScoreLayer2(t).addi(t.getVector().transpose().muli(gradientRate));

                /*
                 * get the child concatenate vector
                 */
                INDArray bias = Nd4j.ones(1, 1);
                INDArray concat = Nd4j.vstack(t.getChild(0).getVector(), bias);

                /*
                 * get and update the transform gradients
                 */
                // (D SCORE/D pre_t) * (D pre_t/D w_t) = (v_l,bias) //the gradient of w_t
                INDArray transformGradients = totalPreoutputGradents.mmul(concat.transpose());
                gradient.getOrInsertUnaryTransform(t).addi(transformGradients);

                /*
                 * back-propagate the gradients to children
                 */
                INDArray leftGradients = leftDerivative(childrenGradients);

                // left child
                INDArray leftNonlinearDerivative = Nd4j.getExecutioner()
                        .execAndReturn(new TanhDerivative(t.getChild(0).getVector().dup()));
                INDArray leftError = leftGradients.mul(leftNonlinearDerivative);
                pairwiseBackProp(t.getChild(0), leftError, gradient, gradientRate);

            } else { // binary branch
//              //tanh（wf * (v_t,v_sum)）* rate  //the gradient of s1
                gradient.getOrInsertBinaryScoreLayer(t).addi(currentTreeNodeNonlinearVector.transpose().muli(gradientRate));
                //why try in binary
                try {
                    //v_t * rate //the gradient of s2
                    gradient.getOrInsertBinaryScoreLayer2(t).addi(t.getVector().transpose().muli(gradientRate));

                } catch (Exception e) {

                    System.err.println("");
                }

                /*
                 * get the child concatenate vector
                 */
                INDArray bias = Nd4j.ones(1, 1);

                INDArray concat = null;

                concat = Nd4j.vstack(
                        t.getChild(0).getVector(),
                        t.getChild(1).getVector(),
                        bias);
                /*
                 * get and update the transform gradients
                 */
                // (D SCORE/D pre_t) * (D pre_t/D w_t) = (v_l,bias) //the gradient of w_t
                INDArray transformGradients = totalPreoutputGradents.mmul(concat.transpose());
                gradient.getOrInsertBinaryTransform(t).addi(transformGradients);

                /*
                 * back-propagate the gradients to children
                 */
                //splt a vector to two
                INDArray leftGradients = leftDerivative(childrenGradients);
                INDArray rightGradients = rightDerivative(childrenGradients);
                INDArray leftNonlinearDerivative = Nd4j.getExecutioner()
                        .execAndReturn(new TanhDerivative(t.getChild(0).getVector().dup()));
                INDArray rightNonlinearDerivative = Nd4j.getExecutioner()
                        .execAndReturn(new TanhDerivative(t.getChild(1).getVector().dup()));
                //左边的梯度乘以tanh'（vt）也就是对左儿子preout的梯度
                INDArray leftError = leftGradients.mul(leftNonlinearDerivative);
                INDArray rightError = rightGradients.mul(rightNonlinearDerivative);

                // left child
                pairwiseBackProp(t.getChild(0), leftError, gradient, gradientRate);

                // right child
                pairwiseBackProp(t.getChild(1), rightError, gradient, gradientRate);

            }
        }
    }

    /**
     * feedforward the CFGRerankingTree neural net and score each non-leaf nodes
     * in the CFGRerankingTree
     *
     * @param t
     * @return
     */
    public double feedForwardAndScore(CFGRerankingTree t) {

        if (t.numChildren() == 0) { //leaf vector, we do not score the leaf node

            INDArray preOutput = this.preOutput(t);
            t.setPreOutput(preOutput);
            INDArray nonLinear = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("tanh", preOutput.dup()));
            t.setVector(nonLinear);
            return 0;

        } else if (t.numChildren() == 1) { //unary

            double childrenScoreSum = this.feedForwardAndScore(t.getChild(0));

            //perform user defined composition
            INDArray preOutput = this.preOutput(t);
            t.setPreOutput(preOutput);
            INDArray nonLinear = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("tanh", preOutput.dup()));
            t.setVector(nonLinear);
            double currentNodeScore = model.getUnaryScoreLayer(t).mmul(nonLinear).getDouble(0);

            return childrenScoreSum + currentNodeScore;
        } else { //binary

            //do these recursive calls in parallel in future
            double leftChildrenScoreSum = this.feedForwardAndScore(t.getChild(0));
            double rightChildrenScoreSum = this.feedForwardAndScore(t.getChild(1));

            //perform user defined composition
            INDArray preOutput = this.preOutput(t);
            t.setPreOutput(preOutput);
            INDArray nonLinear = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("tanh", preOutput.dup()));
            t.setVector(nonLinear);

            double currentNodeScore = model.getBinaryScoreLayer(t).mmul(nonLinear).getDouble(0);

            return currentNodeScore + leftChildrenScoreSum + rightChildrenScoreSum;
        }
    }

    public void backProp(CFGRerankingTree t, INDArray error, SURecursiveNetworkModel gradient) {
        //error term gives us del loss/ del y (y is the output of this node)

        if (t.numChildren() == 0) {
            //fine tune leaf vector embeddings
            gradient.getOrInsertWordVector(t).addi(error);
            return;
        } else {  // children num  ==  1 or 2

            boolean bBinary = t.numChildren() == 2;

            INDArray currentTreeNodeNonlinearVector = t.getVector();
            //nonLinear Derivative = [g'(Wx+b)]
            //Be careful that execAndReturn works on the same copy so duplicate the INDArray
            INDArray currentTreeNodeNonlinearVectorDerivative = Nd4j.getExecutioner()
                    .execAndReturn(new TanhDerivative(currentTreeNodeNonlinearVector.dup()));

            /*
             * compute the gradients according to current node score
             */
            INDArray scoreLayer = bBinary ? model.getBinaryScoreLayer(t) : model.getUnaryScoreLayer(t);
            INDArray transformLayer = bBinary ? model.getBinaryTransform(t) : model.getUnaryTransform(t);

            INDArray currentScoreGradients = scoreLayer.transpose().mul(currentTreeNodeNonlinearVectorDerivative);


            /*
             * get the total gradients of current linear layer of nets
             * the total gradients include gradient from parent : error
             *                         and from the score layer : currentScoreGradients
             */
            INDArray currentTotalGradients = currentScoreGradients.add(error);

            /*
             * get the gradients for children
             */
            INDArray childrenGradients = transformLayer.transpose().mmul(currentTotalGradients);


            /*
             * update the gradients of current CFGRerankingTree node
             */
            if (t.numChildren() == 1) {

                gradient.getOrInsertUnaryScoreLayer(t).addi(currentTreeNodeNonlinearVector.transpose());

                /*
                 * get the child concatenate vector
                 */
                INDArray bias = Nd4j.ones(1, 1);
                INDArray concat = Nd4j.vstack(t.getChild(0).getVector(), bias);

                /*
                 * get and update the transform gradients
                 */
                INDArray transformGradients = currentTotalGradients.mmul(concat.transpose());
                gradient.getOrInsertUnaryTransform(t).addi(transformGradients);

                /*
                 * back-propagate the gradients to children
                 */
                INDArray leftGradients = leftDerivative(childrenGradients);

                // left child
                INDArray leftNonlinearDerivative = Nd4j.getExecutioner()
                        .execAndReturn(new TanhDerivative(t.getChild(0).getVector().dup()));
                INDArray leftError = leftGradients.mul(leftNonlinearDerivative);
                backProp(t.getChild(0), leftError, gradient);

            } else { // binary branch

                gradient.getOrInsertBinaryScoreLayer(t).addi(currentTreeNodeNonlinearVector.transpose());

                /*
                 * get the child concatenate vector
                 */
                INDArray bias = Nd4j.ones(1, 1);
                INDArray concat = Nd4j.vstack(
                        t.getChild(0).getVector(),
                        t.getChild(1).getVector(),
                        bias);
                /*
                 * get and update the transform gradients
                 */
                INDArray transformGradients = currentTotalGradients.mmul(concat.transpose());
                gradient.getOrInsertBinaryTransform(t).addi(transformGradients);

                /*
                 * back-propagate the gradients to children
                 */
                INDArray leftGradients = leftDerivative(childrenGradients);
                INDArray rightGradients = rightDerivative(childrenGradients);
                INDArray leftNonlinearDerivative = Nd4j.getExecutioner()
                        .execAndReturn(new TanhDerivative(t.getChild(0).getVector().dup()));
                INDArray rightNonlinearDerivative = Nd4j.getExecutioner()
                        .execAndReturn(new TanhDerivative(t.getChild(1).getVector().dup()));

                INDArray leftError = leftGradients.mul(leftNonlinearDerivative);
                INDArray rightError = rightGradients.mul(rightNonlinearDerivative);

                // left child
                backProp(t.getChild(0), leftError, gradient);

                // right child
                backProp(t.getChild(1), rightError, gradient);

            }
        }
    }

    public INDArray leftDerivative(INDArray childrenGradients) {

        INDArray leftDerivative = Nd4j.zeros(this.n, 1);
        for (int row = 0; row < this.n; row++) {
            leftDerivative.putRow(row, childrenGradients.getRow(row));
        }

        return leftDerivative;
    }

    public INDArray rightDerivative(INDArray childrenGradients) {

        INDArray rightDerivative = Nd4j.zeros(this.n, 1);
        for (int row = this.n; row < 2 * this.n; row++) {
            rightDerivative.putRow(row - this.n, childrenGradients.getRow(row));
        }

        return rightDerivative;
    }

    public int getDimension() {
        return this.n;
    }

    public double getScore(CFGRerankingTree CFGRerankingTree) {

        return feedForwardAndScore(CFGRerankingTree);
    }

    public void updateGradients(SURecursiveNetworkModel gradient, IdentityHashMap<INDArray, AdaGrad> gradientSquareMap,
                                int batchSize, double fRegRate) {
        model.updateModel(gradient, gradientSquareMap, batchSize, fRegRate);
    }

    /**
     * pair wised compare the given two trees, and return whether
     * the first tree is better than the second
     *
     * @param tree1
     * @param tree2
     * @return
     */
    public boolean pairwiseCompare(CFGRerankingTree tree1, CFGRerankingTree tree2) {

        List<CFGRerankingTree> tree1List = new ArrayList<>();
        List<CFGRerankingTree> tree2List = new ArrayList<>();

        // first stage forward
        firstStageForward(tree1, tree1List);
        firstStageForward(tree2, tree2List);

        // get the tree lattice
        TreeLattice tree1Lattice = new TreeLattice(tree1List);
        TreeLattice tree2Lattice = new TreeLattice(tree2List);

        // second stage pair-wised forward
        double tree1Score = secondStageForward(tree1, tree2Lattice);
        double tree2Score = secondStageForward(tree2, tree1Lattice);

        return tree1Score >= tree2Score;

    }

    /**
     * get two stage score for pair-wise reranking
     *
     * @param tree1
     * @param tree2
     * @return
     */
    public Pair<Double, Double> pairwiseGetScore(CFGRerankingTree tree1, CFGRerankingTree tree2) {

        List<CFGRerankingTree> tree1List = new ArrayList<>();
        List<CFGRerankingTree> tree2List = new ArrayList<>();

        // first stage forward
        double tree1FirstStageScore = firstStageForward(tree1, tree1List);
        double tree2FirstStageScore = firstStageForward(tree2, tree2List);

        // get the tree lattice
        TreeLattice tree1Lattice = new TreeLattice(tree1List);
        TreeLattice tree2Lattice = new TreeLattice(tree2List);

        // second stage pair-wised forward
        double tree1Score = secondStageForward(tree1, tree2Lattice);
        double tree2Score = secondStageForward(tree2, tree1Lattice);

        return Pair.makePair(tree1Score + tree1FirstStageScore, tree2Score + tree2FirstStageScore);

    }


}
