/* =================================================================================
 *
 * D-Cube: Dense-Block Detection in Terabyte-Scale Tensors
 * Authors: Kijung Shin, Bryan Hooi, Jisu Kim, and Christos Faloutsos
 *
 * Version: 1.0
 * Date: August 6, 2016
 * Main Contact: Kijung Shin (kijungs@cs.cmu.edu)
 *
 * This software is free of charge under research purposes.
 * For commercial purposes, please contact the author.
 *
 * =================================================================================
 */

package dcube;

import java.io.*;

/**
 * Order by which attribute values are removed
 * @author kijungs
 */
public class BlockIterInfo {

    private int dimension = 0;
    private byte[] modes = null;
    private int[] attributes = null;
    private boolean useBuffer = true;
    private int cardinalitySum = 0;
    private int curIndex = 0;
    private ObjectOutputStream out = null;
    private String orderingFilePath = null;

    public static boolean properBufferUsage(int modeLengthSum) {
        long memoryToUse = Runtime.getRuntime().maxMemory() * 7 / 10;
        System.gc();
        long memoryUsed = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
        long memoryLeft = memoryToUse - memoryUsed;
        if(memoryLeft > (Integer.BYTES + Byte.BYTES) * (long) modeLengthSum) {
            return true;
        }
        else {
            return false;
        }
    }

    public BlockIterInfo(int[] modeLengths, boolean useBuffer, String orderingFilePath) throws IOException {
        // System.out.println("useBuffer: " + useBuffer);
        this.useBuffer = useBuffer;
        this.dimension = modeLengths.length;
        for(int mode = 0; mode < dimension; mode++) {
            cardinalitySum += modeLengths[mode];
        }
        this.modes = new byte[cardinalitySum];
        this.attributes = new int[cardinalitySum];
        if(!useBuffer) {
            this.orderingFilePath = orderingFilePath;
            out = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream(orderingFilePath), 8388608));
        }
    }

    public void addIterInfo(byte mode, int index) throws IOException {
        if(useBuffer) {
            modes[curIndex] = mode;
            attributes[curIndex++] = index;
        }
        else {
            out.writeByte(mode);
            out.writeInt(index);
        }
    }

    public BlockInfo returnBlock(int maxIter, String blockInfoPath) throws IOException {
        if (out != null) {
            out.close();
        }

        if(useBuffer) { // write block info in memory
            int[] modeLengths = new int[dimension];
            int newLength = cardinalitySum - maxIter;
            byte[] newModes = new byte[newLength];
            for(int i = 0; i < newLength; i++) {
                newModes[i] = modes[i+maxIter];
                modeLengths[modes[i+maxIter]]++;
            }
            int[] newIndices = new int[newLength];
            for(int i = 0; i < newLength; i++) {
                newIndices[i] = attributes[i+maxIter];
            }
            return new BlockInfo(newLength, modeLengths, newModes, newIndices);
        }
        else { //write block info in disk

            int[] modeLengths = new int[dimension];
            int newLength = cardinalitySum - maxIter;
            ObjectOutputStream out = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream(blockInfoPath), 8388608));
            ObjectInputStream in = new ObjectInputStream(new BufferedInputStream(new FileInputStream(orderingFilePath), 8388608));

            for(int i = 0; i < maxIter; i++) { //throw away
                in.readByte();
                in.readInt();
            }
            for(int i = 0; i < newLength; i++) {
                byte mode = in.readByte();
                out.writeByte(mode);
                modeLengths[mode]++;
                out.writeInt(in.readInt());
            }
            in.close();
            out.close();

            if(new File(orderingFilePath).exists()) {
                new File(orderingFilePath).delete();
            }

            return new BlockInfo(newLength, modeLengths, blockInfoPath);
        }
    }
}
