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
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.HashSet;
import java.util.Set;


/**
 * Information about each block found
 * @author kijungs
 */
public class BlockInfo {

    public int[] blockCardinalities;
    private byte[] modes = null;
    private int[] attVals = null;
    private int size = 0;
    public String diskFilePath = null;
    private boolean useBuffer = false;

    public BlockInfo(int size, int[] blockCardinalities, byte[] modes, int[] attributes) {
        this.size = size;
        this.useBuffer = true;
        this.modes = modes;
        this.attVals = attributes;
        this.blockCardinalities = blockCardinalities;
    }

    public BlockInfo(int size, int[] blockCardinalities, String diskFilePath) {
        this.size = size;
        this.useBuffer = false;
        this.diskFilePath = diskFilePath;
        this.blockCardinalities = blockCardinalities;
    }

    public Set<Integer>[] getAttributeValues(int dimension) throws IOException {
        Set<Integer>[] modeToAttVals = new Set[dimension];
        for(int mode = 0; mode < dimension; mode++) {
            modeToAttVals[mode] = new HashSet<Integer>();
        }
        if(useBuffer) {
            for (int i = 0; i < size; i++) {
                byte mode = modes[i];
                modeToAttVals[mode].add(attVals[i]);
            }
        }
        else {
            ObjectInputStream in = new ObjectInputStream(new BufferedInputStream(new FileInputStream(diskFilePath), 8388608));
            for(int i = 0; i < size; i++) {
                byte mode = in.readByte();
                modeToAttVals[mode].add(in.readInt());
            }
            in.close();
        }
        return modeToAttVals;
    }
    
    public boolean[][] getBitMask(int dimension, int[] cardinalities) throws IOException {

        final boolean[][] modeToIndexToBeingIncluded = new boolean[dimension][];
        for(int mode = 0; mode < dimension; mode++) {
            modeToIndexToBeingIncluded[mode] = new boolean[cardinalities[mode]];
        }

        if(useBuffer) {
            for (int i = 0; i < size; i++) {
                byte mode = modes[i];
                modeToIndexToBeingIncluded[mode][attVals[i]] = true;
            }
        }
        else {
            ObjectInputStream in = new ObjectInputStream(new BufferedInputStream(new FileInputStream(diskFilePath), 8388608));
            for(int i = 0; i < size; i++) {
                byte mode = in.readByte();
                modeToIndexToBeingIncluded[mode][in.readInt()] = true;
            }
            in.close();
        }

        return modeToIndexToBeingIncluded;
    }

    public String returnFileInfo(String tempLocalFilePath) throws IOException {
        if(useBuffer) {
            String newPath = tempLocalFilePath;
            ObjectOutputStream out = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream(newPath), 8388608));
            for(int i = 0; i < size; i++) {
                out.writeByte(modes[i]);
                out.writeInt(attVals[i]);
            }
            out.close();
            return newPath;
        }
        else {
            String newPath = tempLocalFilePath;
            Files.copy(new File(diskFilePath).toPath(), new File(newPath).toPath(), StandardCopyOption.REPLACE_EXISTING);
            return newPath;
        }

    }

    public void clear() {
        if(!useBuffer) {
            if (new File(diskFilePath).exists()) {
                new File(diskFilePath).delete();
            }
        }
    }

}
