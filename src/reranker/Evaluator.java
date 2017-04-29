package reranker;

import edu.berkeley.nlp.syntax.Tree;
import edu.berkeley.nlp.syntax.Trees;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.trees.PennTreebankLanguagePack;
import edu.stanford.nlp.trees.TreebankLanguagePack;
import nncon.CFGTree;
import nncon.EnglishPennTreebankParseEvaluator;
import nncon.Evalb;

import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringReader;
import java.util.*;

/**
 * Created by zhouh on 15-12-15.
 *
 */
public class Evaluator {

    /**
     * evalb for sentence
     */
    EnglishPennTreebankParseEvaluator.LabeledConstituentEval<String> eval = null;
    private final TreebankLanguagePack tlp;

    public Evaluator(TreebankLanguagePack tlp) {
        eval = new EnglishPennTreebankParseEvaluator.LabeledConstituentEval<String>(
                new HashSet<String>(Arrays.asList(new String[]{"ROOT"})),
                new HashSet<String>());

        this.tlp = tlp;

    }

    /**
     * Evaluate performance on a list of sentences, predicted parses,
     * and gold parses.
     *
     * @return A map from metric name to metric value
     * @throws IOException
     */
    public double evaluate(List<CFGRerankingTree> trees,
                           List<CFGRerankingTree> goldTrees) throws IOException {

        PrintWriter outguess = IOUtils.getPrintWriter("guessTrees");
        PrintWriter outgold = IOUtils.getPrintWriter("goldTrees");

        for (CFGRerankingTree guesstree : trees){
            outguess.println(guesstree.getStr());

        }
        outguess.close();

        for (CFGRerankingTree goldtree : goldTrees)
            outgold.println(goldtree.getStr());
        outgold.close();

        if (tlp instanceof PennTreebankLanguagePack) {
            String[] path = {"goldTrees", "guessTrees"};
            return Evalb.massEvalb(path);
        } else {
            String[] path = {"goldTrees", "guessTrees", "-l Chinese"};
            return Evalb.massEvalb(path);
        }
    }

    public double evaluateOracle4CFGRerankingTree(List<List<RerankCandidate>> kbest, List<CFGRerankingTree> goldTrees) throws IOException {

        // oracle computing
        List<CFGRerankingTree> bestTrees = new ArrayList<>();

        //output
        for (int j = 0; j < goldTrees.size(); j++) {

            double bestF1 = Double.NEGATIVE_INFINITY;
            CFGRerankingTree bestTree = null;

            List<RerankCandidate> nbest = kbest.get(j);
            String goldTreeStr = goldTrees.get(j).getStr();
            //the first one of returns

            for (int k = 0; k < nbest.size(); k++) {

                //get oracle
                double f1 = evaluateSent(nbest.get(k).tree.getStr(), goldTreeStr);
                if (f1 > bestF1 || bestTree == null) {
                    bestF1 = f1;
                    bestTree = nbest.get(k).tree;
                }

            }
            bestTrees.add(bestTree);
        }

        return evaluate(bestTrees, goldTrees);
    }


    /**
     * Evaluate performance on a list of sentences, predicted parses,
     * and gold parses.
     *
     * @return A map from metric name to metric value
     * @throws IOException
     */
    public double evaluate(List<String> sentences, List<CFGTree> trees,
                           List<CFGTree> goldTrees) throws IOException {

        PrintWriter outguess = IOUtils.getPrintWriter("guessTrees");
        PrintWriter outgold = IOUtils.getPrintWriter("goldTrees");

        for (CFGTree guesstree : trees)
            outguess.println(guesstree.toString());
        outguess.close();

        for (CFGTree goldtree : goldTrees)
            outgold.println(goldtree.toString());
        outgold.close();

        if (tlp instanceof PennTreebankLanguagePack) {
            String[] path = {"goldTrees", "guessTrees"};
            return Evalb.massEvalb(path);
        } else {
            String[] path = {"goldTrees", "guessTrees", "-l Chinese"};
            return Evalb.massEvalb(path);
        }


    }

    public double evaluateFile(String guessFileName, String goldFileName) throws IOException {


        if (tlp instanceof PennTreebankLanguagePack) {
            String[] path = {goldFileName, guessFileName};
            return Evalb.massEvalb(path);
        } else {
            String[] path = {goldFileName, guessFileName, "-l Chinese"};
            return Evalb.massEvalb(path);
        }


    }



    /**
     * TODO: we need to compare the two results of evaluation
     *
     * @param treeStr
     * @param goldTreeStr
     * @return
     * @throws IOException
     */
    public double evaluateSent(String treeStr, String goldTreeStr) throws IOException {

        Tree<String> guessedTree = (new Trees.PennTreeReader(new StringReader(
                treeStr))).next();
        Tree<String> goldTree = (new Trees.PennTreeReader(new StringReader(
                goldTreeStr))).next();

        return eval.evaluateAndReturnPara(guessedTree, goldTree, null);
    }


    public double evaluateOracle(List<List<CFGTree>> kbest, List<CFGTree> goldTrees) throws IOException {

        // oracle computing
        double bestF1 = 0;
        CFGTree bestTree = null;
        List<CFGTree> bestTrees = new ArrayList<>();

        //output
        for (int j = 0; j < goldTrees.size(); j++) {

            List<CFGTree> nbest = kbest.get(j);
            String goldTreeStr = goldTrees.get(j).toString();
            //the first one of returns

            for (int k = 0; k < nbest.size(); k++) {

                //get oracle
                double f1 = evaluateSent(nbest.get(k).toString(), goldTreeStr);
                if (f1 > bestF1 || bestTree == null) {
                    bestF1 = f1;
                    bestTree = nbest.get(k);
                }

            }
            bestTrees.add(bestTree);
        }

        return evaluate(null, bestTrees, goldTrees);
    }


}
