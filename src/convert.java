import nncon.CFGTree;
import nncon.CFGTreeNode;
import nndep.Util;

import java.io.*;

import static nncon.CFGTree.getTokens;


public class convert {
    public static void main(String[] args) throws Exception{
        BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream("/Users/qiwang/data/small.gold")));
        BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("/Users/qiwang/data/small.gold2")));
        CFGTree tree = new CFGTree();
        while (true) {
            String line = reader.readLine();
            if (line == null) break;
            if (line.isEmpty()) {
                writer.write(line);
            }else if (line.trim().charAt(0) == '(') {
                tree.CTBReadNote(getTokens(line));
                String newstr = tree.toString();
                writer.write(newstr+"\r\n");
            }
        }
        writer.close();
        reader.close();
    }
}
