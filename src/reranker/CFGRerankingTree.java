package reranker;

import nncon.CFGTree;
import nncon.CFGTreeNode;
import org.nd4j.linalg.api.ndarray.INDArray;
import recursivenetwork.abstractnetwork.Tree;
import recursivenetwork.util.SumOfGradient;

import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

/**
 * Created by zhouh on 15-12-10.
 *
 */
public class CFGRerankingTree {

    /**
     * for leaves this contains reference to a global leaf vector
     */
    private INDArray vector;
    private INDArray gatedVector;
    /**
     * INDArray before non-linearity is applied
     */
    private INDArray preOutput;
    /**
     * gradient is non-null only for leaf nodes. Contains reference. Contains
     * reference to the unique gradient vector corresponding to the leaf
     */
    private SumOfGradient gradient;
    private final String label;
    public final LinkedList<CFGRerankingTree> children;
    private final int numChild;
    private String head;
    private boolean bTemp;
    public INDArray concatVector4Gate;

    public INDArray pairWiseGradient;
    public List<CFGRerankingTree> treeCollections; // It is only not null when this is the root tree
    public List<CFGRerankingTree> gateSpamTree;
    public double[] gateWeights;

    public int begin;
    public int end;

    /**
     * @param label
     * @param children
     */
    public CFGRerankingTree(String label, LinkedList<CFGRerankingTree> children, String head, boolean bTemp) {
        this.label = label;
        this.children = children;
        this.numChild = children.size();
        this.head = head;
        this.bTemp = bTemp;
    }

    public void clear(){
        gatedVector = null;
        concatVector4Gate = null;
        pairWiseGradient = null;
        treeCollections = null;
        gateSpamTree = null;
        gateWeights = null;

        vector = null;
        preOutput = null;

        for (CFGRerankingTree child : children) {
            child.clear();
        }
    }

    public void setConcatVector4Gate(INDArray concatVector4Gate) {
        this.concatVector4Gate = concatVector4Gate;
    }

    public void setGatedVector(INDArray gatedVector) {
        this.gatedVector = gatedVector;
    }

    public INDArray getGatedVector() {
        return gatedVector;
    }

    public void setTreeCollections(List<CFGRerankingTree> treeCollections) {
        this.treeCollections = treeCollections;
    }

    public void setGateSpamTree(List<CFGRerankingTree> gateSpamTree) {
        this.gateSpamTree = gateSpamTree;
    }

    public void setGateWeights(double[] gateWeights) {
        this.gateWeights = gateWeights;
    }

    public void setPairWiseGradient(INDArray pairWiseGradient) {
        if (this.pairWiseGradient == null)
            this.pairWiseGradient = pairWiseGradient;
        else
            this.pairWiseGradient.addi( pairWiseGradient );
    }

    public void clearPairWiseGradient(){
        this.pairWiseGradient = null;
    }

    /**
     * Convert the CFGTree~(used in the baseline parser) to
     * Reranking tree
     *
     * @return
     */
    public static CFGRerankingTree CFGTree2RerankingTree(CFGTree cfgTree) {

        /*
         * get the root node of a constituent binary tree
         */
        CFGTreeNode cfgTreeRootNode = cfgTree.nodes.get(cfgTree.nodes.size() - 1);
        CFGRerankingTree rootRerankingTree = CFGTreeNode2RerankingTree(cfgTree, cfgTreeRootNode);
        return rootRerankingTree;
    }

    /**
     * Convert the CFGTreeNode~(used in the baseline parser) to
     * Reranking tree node
     *
     * @return
     */
    public static CFGRerankingTree CFGTreeNode2RerankingTree(CFGTree cfgTree, CFGTreeNode cfgNode) {

        if (!cfgNode.is_constituent) {
            CFGRerankingTree retval = new CFGRerankingTree(cfgNode.constituent, new LinkedList<>(),
                    cfgNode.word, false);
            return retval;
        } else {
            LinkedList<CFGRerankingTree> chidlren = new LinkedList<>();
            CFGRerankingTree leftChild = CFGTreeNode2RerankingTree(cfgTree, cfgTree.getLeftChild(cfgNode));
            chidlren.add(leftChild);
            if (!cfgNode.single_child) {
                CFGRerankingTree rightChild =
                        CFGTreeNode2RerankingTree(cfgTree, cfgTree.getRightChild(cfgNode));
                chidlren.add(rightChild);
            }
            CFGRerankingTree retval = new CFGRerankingTree(cfgNode.constituent, chidlren,
                    cfgNode.getHead(cfgTree), cfgNode.temp);

            return retval;
        }
    }

    public String getWord() {
        return head;
    }

    public int numChildren() {
        return this.numChild;
    }

    public CFGRerankingTree getChild(int i) {
        /* use iterator for trees with large degree */
        return this.children.get(i);
    }

    public Iterator<CFGRerankingTree> getChildren() {
        return this.children.iterator();
    }

    public INDArray getVector() {
        return this.vector;
    }

    public void setVector(INDArray vector) {
        this.vector = vector;
    }

    public SumOfGradient getGradient() {
        return this.gradient;
    }

    public void setGradient(SumOfGradient sumOfGradient) {
        this.gradient = sumOfGradient;
    }

    /**
     * accumulate gradient so that after a set of backprop through several trees
     * you can update all the leaf vectors by the sum of gradients. Don't forget to
     * clear the gradients after updating.
     */
    public void addGradient(INDArray gradient) {
        this.gradient.addGradient(gradient);
    }

    public String getLabel() {
        return this.label;
    }

    public INDArray getPreOutput() {
        return this.preOutput;
    }

    public void setPreOutput(INDArray preOutput) {
        this.preOutput = preOutput;
    }

    public String getStr() {

        String retval = "";

        if (bTemp) {
            retval += children.get(0).getStr();
            retval += " ";
            retval += children.get(1).getStr();
        } else {
            retval += "(" + label + " ";

            if (numChild == 0) {

                retval += head + ")";
                return retval;
            } else {
                if (numChild >= 1) retval += children.get(0).getStr();
                retval += " ";
                if (numChild >= 2) retval += children.get(1).getStr();
                retval += ")";
            }
        }

        return retval;
    }

    public CFGRerankingTree copy() {

        CFGRerankingTree tree = new CFGRerankingTree(label, new LinkedList<>(), head, bTemp);


        for (int i = 0; i < this.children.size(); i++) {
            tree.children.add( (children.get(i).copy()) );
        }

        return tree;
    }

}
