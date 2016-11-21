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
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

/**
 * D-Cube Implementation
 * @author kijungs
 */
public class Proposed {

    /**
     * Main function
     *
     * @param args input_path, output_path, num_of_attributes, density_measure, num_of_blocks
     * @throws IOException
     */
    public static void main(String[] args) throws IOException {
        if (args.length < 5) {
            printError();
            System.exit(-1);
        }

        final String input = args[0];
        System.out.println("input_path: " + input);

        final String output = args[1];
        System.out.println("output_path: " + output);
        File dir = new File(output);
        try{
            dir.mkdir();
        }
        catch(Exception e){
        }

        final int dimension = Integer.valueOf(args[2]);
        System.out.println("dimension: " + dimension);

        DensityMeasure densityMeasure = null;
        if (args[3].compareToIgnoreCase("ARI") == 0) {
            densityMeasure = DensityMeasure.Arithmetic;
        } else if (args[3].compareToIgnoreCase("GEO") == 0) {
            densityMeasure = DensityMeasure.Geometric;
        } else if (args[3].compareToIgnoreCase("SUSP") == 0) {
            densityMeasure = DensityMeasure.Suspiciousness;
        } else {
            System.err.println("Unknown Density Measure");
            printError();
            System.exit(-1);
        }
        System.out.println("density_measure: " + args[3]);

        int policy = 0;
        if (args[4].compareToIgnoreCase("CARDINALITY") == 0) {
            policy = POLICY_MAX_CARNDILITY;
        } else if(args[4].compareToIgnoreCase("DENSITY") == 0) {
            policy = POLICY_MAX_DENSITY;
        } else {
            System.err.println("Unknown Policy");
            printError();
            System.exit(-1);
        }
        System.out.println("policy: " + args[4]);
        
        final int blockNum = Integer.valueOf(args[5]);
        System.out.println("num_of_blocks: " + blockNum);

        System.out.println();
        System.out.println("computing proper buffer size");
        Pair<Long, int[]> info = probe(dimension, input, ",");
        long omega = info.getKey();
        int[] cardinalities = info.getValue();
        int bufferSize = getProperBufferSizeForInputTensor(dimension, omega, cardinalities);

        System.out.println();
        System.out.println("storing the input tensor in the binary format...");
        Tensor tensor = TensorMethods.importSparseTensor(input, ",", dimension, cardinalities, bufferSize, getFullPath(output, Proposed.originalAttName), getFullPath(output, Proposed.originalValueName));

        System.out.println();
        System.out.println("running the algorithm...");
        Proposed proposed = new Proposed(tensor, output);
        System.out.println();
        proposed.run(blockNum, densityMeasure, policy);
    }

    private static void printError() {
        System.err.println("Usage: run_single.sh input_path output_path dimension density_measure policy num_of_blocks");
        System.err.println("Density_measure should be one of [ari, geo, susp]");
        System.err.println("Policy should be one of [density, cardinality]");
    }


    public final static String originalAttName = "disk_att_original";
    public final static String originalValueName = "disk_value_original";
    public final static String currentValueName = "disk_value_current";
    public final static String blockAttName = "disk_att_block";
    public final static String blockValueName = "disk_value_block";
    public final static String tempAttName = "disk_att_temp";
    public final static String tempValueName = "disk_value_temp";

    private String outputPath = "";

    private Tensor Rori;
    private Tensor R;
    private Tensor B;

    private int[][] attValMasses;

    protected enum TensorType{
        OriginalR, CurrentR
    }

    public static final int POLICY_MAX_CARNDILITY = 0;
    public static final int POLICY_MAX_DENSITY = 1;


    public Proposed(Tensor tensor, String outputPath) throws IOException {
        Rori = tensor;
        this.outputPath = outputPath;
    }

    private static String getFullPath(String outputPath, String fileName) {
        return outputPath + File.separator + fileName;
    }
    
    private String getFullPath(String fileName) {
        return outputPath + File.separator + fileName;
    }

    private String getOrderingFullPath(int blockIndex) {
        return outputPath + File.separator + "ordering_info" + blockIndex;
    }

    private String getBlockInfoFullPath(int blockIndex) {
        return outputPath + File.separator + "block_info" + blockIndex;
    }


    /**
     * get statistics of the input tensor, which are used to decide where to store the input data
     * @param dimension dimension of the input tensor
     * @param path path to the input tensor
     * @param delim delimeter used in the input tensor
     * @return (omega, cardinalities)
     */
    public static Pair<Long, int[]> probe(int dimension, String path, String delim) throws IOException {

        long omega = 0; // number of observable entries
        final int[] maxAttVals = new int[dimension];

        final BufferedReader br = new BufferedReader(new FileReader(path));
        while(true){
            final String line = br.readLine();
            if(line==null)
                break;

            final String[] tokens = line.split(delim);
            if(tokens.length < dimension + 1) {
                System.out.println("Skipped Line: " + line);
                continue;
            }
            omega++;
            for(int mode = 0; mode < dimension; mode++) {
                maxAttVals[mode] = Math.max(maxAttVals[mode], Integer.valueOf(tokens[mode]));
            }
        }
        br.close();
        final int[] cardinalities = new int[dimension];
        for(int mode = 0; mode < dimension; mode++) {
            cardinalities[mode] = maxAttVals[mode] + 1;
        }

        return new Pair(omega, cardinalities);
    }

    /**
     * get the propose size of the buffer for the input tensor (R and Rori)
     * @param dimension dimension of the input tensor
     * @param omega number of tuples in the input tensor
     * @param cardinalities n -> cardinality of the n-th attribute
     * @return propoer size of the buffer for the input tensor
     */
    public static int getProperBufferSizeForInputTensor(int dimension, long omega, int[] cardinalities){

        long memoryToUse = Runtime.getRuntime().maxMemory() * 7 / 10;
        long cardinalitiesum = 0;
        for(int mode=0; mode<dimension; mode++) {
            cardinalitiesum = cardinalities[mode];
        }
        System.gc();
        long memoryRequired = omega * (2 * (dimension+1) + 1) * Integer.BYTES + 4 * cardinalitiesum * Integer.BYTES ;
        long memoryUsed = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
        if(memoryToUse > memoryRequired + memoryUsed && omega < Integer.MAX_VALUE) {
            return (int)omega;
        }
        else {
            return 0;
        }
    }

    /**
     * get the propose size of the buffer for the current block (B)
     * @param dimension dimension of the input tensor
     * @return propoer size of the buffer for the current block (B)
     */
    private long getProperBufferSizeForBlocks(int dimension){

        long memoryToUse = Runtime.getRuntime().maxMemory() * 7 / 10;
        System.gc();
        long memoryUsed = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
        long memoryLeft = memoryToUse - memoryUsed;
        return Math.max(0L, memoryLeft / Integer.BYTES / (dimension+1));
    }

    /**
     * get the mass of the given relation
     * @return
     */
    private long getMass(TensorType type) {
        if(type == TensorType.OriginalR) {
            return Rori.mass;
        }
        else {
            return R.mass;
        }
    }

    /**
     * get the attribute-value mass
     * @return
     */
    private int[][] getAttValMasses() {
        return attValMasses;
    }

    private int getDimension() {
        return Rori.dimension;
    }

    private int[] getCardinalities() {
        return Rori.cardinalities;
    }

    /**
     * copy the original tensor
     */
    private void copyOriTesnor() throws IOException {
        R = Rori.copy(getFullPath(currentValueName));
    }

    /**
     * Copy the current block
     */
    private void copyBlock(){
        B = null;
        int bufferSize = (int)(Math.min(Integer.MAX_VALUE, Math.min(getProperBufferSizeForBlocks(R.dimension), R.bufferUsage + R.diskUsage)));
        B = new Tensor(R.dimension, R.cardinalities, R.bufferUsage + R.diskUsage, R.mass, bufferSize, getFullPath(blockAttName), getFullPath(blockValueName));
    }

    /**
     * Compute the attribute-value masses
     */
    protected void initialize() throws IOException {
        attValMasses = null;
        attValMasses = TensorMethods.attributeValueMasses(R);
    }

    /**
     * Remove tuples and update attribute-value masses
     * @param modeToRemove mode from which attribute values are removed
     * @param attToRemove i -> whether a_{i} should be removed or not
     * @param attMasses attribute-value masses
     * @param isFirst true if this is the first removal false otherwise
     * @throws IOException
     */
    protected void removeAndUpdateAttValMasses(int modeToRemove, boolean[] attToRemove, int[][] attMasses, boolean isFirst) throws IOException {

        int dimension = B.dimension;
        int[][] attributes = B.attributes;
        int[] modeAttributes = B.attributes[modeToRemove];
        int[] values = B.measureValues;
        int bufferSize = B.bufferSize;

        for(int mode = 0; mode < dimension; mode++) {
            if(mode == modeToRemove)
                continue;
            int bufferUsage = B.bufferUsage;
            int[] modeDegree = attMasses[mode];
            int[] updatedModeAttributes = attributes[mode];
            for (int bufferIndex = 0; bufferIndex < bufferUsage; bufferIndex++) {
                if (attToRemove[modeAttributes[bufferIndex]]) {
                    modeDegree[updatedModeAttributes[bufferIndex]] -= values[bufferIndex];
                    bufferUsage--;
                    for(; bufferUsage > bufferIndex; bufferUsage--){
                        if(!attToRemove[modeAttributes[bufferUsage]]) {
                            updatedModeAttributes[bufferIndex] = updatedModeAttributes[bufferUsage];
                            break;
                        }
                        else {
                            modeDegree[updatedModeAttributes[bufferUsage]] -= values[bufferUsage];
                        }
                    }
                }
            }
        }
        int bufferUsage = B.bufferUsage;
        int[] modeAttMasses = attMasses[modeToRemove];
        for (int bufferIndex = 0; bufferIndex < bufferUsage; bufferIndex++) {
            if (attToRemove[modeAttributes[bufferIndex]]) {
                modeAttMasses[modeAttributes[bufferIndex]] = 0;  // set to 0;
                bufferUsage--;
                for(; bufferUsage > bufferIndex; bufferUsage--){
                    if(!attToRemove[modeAttributes[bufferUsage]]) {
                        modeAttributes[bufferIndex] = modeAttributes[bufferUsage];
                        values[bufferIndex] = values[bufferUsage];
                        break;
                    }
                    else {
                        modeAttMasses[modeAttributes[bufferUsage]] = 0;
                    }
                }
            }
        }
        B.bufferUsage = bufferUsage;
        int bufferIndex = bufferUsage;


        if(B.diskUsage == 0) {
            B.bufferUsage = bufferIndex;
            return;
        }

        if(isFirst) { //read from current Tensor

            int[] curTensorValues = R.measureValues;
            int[] curTensorModeAttributes = R.attributes[modeToRemove];
            final int curTensorBufferUsage =  R.bufferUsage;
            for (int mode = 0; mode < dimension; mode++) {
                if(mode == modeToRemove)
                    continue;
                int tempBufferIndex = bufferIndex;
                int[] updatedModeIndices = attributes[mode];
                modeAttMasses = attMasses[mode];
                int[] curTensorUpdatedModeIndices = R.attributes[mode];
                for (int i = 0; i < curTensorBufferUsage; i++) {
                    if(attToRemove[curTensorModeAttributes[i]]) {
                        modeAttMasses[curTensorUpdatedModeIndices[i]] -= curTensorValues[i];
                    }
                    else {
                        updatedModeIndices[tempBufferIndex++] = curTensorUpdatedModeIndices[i];
                    }
                }
            }

            modeAttMasses = attMasses[modeToRemove];
            for (int i = 0; i < curTensorBufferUsage; i++) {
                if(attToRemove[curTensorModeAttributes[i]]) {
                    modeAttMasses[curTensorModeAttributes[i]] -= curTensorValues[i]; // set to 0
                }
                else {
                    modeAttributes[bufferIndex] = curTensorModeAttributes[i];
                    values[bufferIndex++] = curTensorValues[i];
                }
            }

        }

        long newDiskUsage = 0;
        Tensor inputTensor = isFirst ? R : B;

        if(inputTensor.diskUsage > 0) {

            ObjectOutputStream outAtt = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream(getFullPath(tempAttName)), 8388608));
            ObjectOutputStream outValue = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream(getFullPath(tempValueName)), 8388608));

            ObjectInputStream inAtt = new ObjectInputStream(new BufferedInputStream(new FileInputStream(isFirst ? R.attFilePath : B.attFilePath), 8388608));
            ObjectInputStream inValue = new ObjectInputStream(new BufferedInputStream(new FileInputStream(isFirst ? R.valueFilePath : B.valueFilePath), 8388608));

            int[] tupleAttValues = new int[dimension];
            while (true) {
                int value = inValue.readInt();
                if (value == -1) {
                    outValue.writeInt(-1);
                    break;
                } else if (value == 0) { //already removed entry
                    for (int mode = 0; mode < dimension; mode++) {
                        inAtt.readInt();
                    }
                    continue;
                }

                for (int mode = 0; mode < dimension; mode++) {
                    tupleAttValues[mode] = inAtt.readInt();
                }
                if (attToRemove[tupleAttValues[modeToRemove]]) {
                    for (int mode = 0; mode < dimension; mode++) {
                        attMasses[mode][tupleAttValues[mode]] -= value;
                    }
                } else {
                    if (bufferIndex < bufferSize) {
                        for (int mode = 0; mode < dimension; mode++) {
                            attributes[mode][bufferIndex] = tupleAttValues[mode];
                        }
                        values[bufferIndex] = value;
                        bufferIndex++;
                    } else {
                        newDiskUsage++;
                        outValue.writeInt(value);
                        for (int mode = 0; mode < dimension; mode++) {
                            outAtt.writeInt(tupleAttValues[mode]);
                        }
                    }
                }
            }
            inAtt.close();
            inValue.close();
            outAtt.close();
            outValue.close();
        }

        B.bufferUsage = bufferIndex;
        B.diskUsage = newDiskUsage;

        if(newDiskUsage > 0 ) {
            new File(getFullPath(blockAttName)).delete();
            new File(getFullPath(blockValueName)).delete();
            new File(getFullPath(tempAttName)).renameTo(new File(getFullPath(blockAttName)));
            new File(getFullPath(tempValueName)).renameTo(new File(getFullPath(blockValueName)));
        }

    }

    protected double removeAndEvaluateBlock(int blockIndex, BlockInfo block, IDensityMeasure measure) throws IOException {

        final int dimension = Rori.dimension;
        long massB = 0;

        int[] cardinalities = getCardinalities();
        for(int mode = 0; mode < dimension; mode++) {
            attValMasses[mode] = new int[cardinalities[mode]];
        }

        final boolean[][] modeToindicesToRemoveArr = block.getBitMask(dimension, cardinalities);
        final int[] cardinalitiesOfBlock = block.blockCardinalities;

        final int[][] attributes = R.attributes;
        final int[] oriValues = Rori.measureValues;
        final int[] values = R.measureValues;
        final int bufferUsage = R.bufferUsage;
        for(int i = 0; i< bufferUsage; i++) {
            boolean removed = true;
            for (int mode = 0; mode < dimension; mode++) {
                if (!modeToindicesToRemoveArr[mode][attributes[mode][i]]) {
                    removed = false;
                    break;
                }
            }

            if(removed) {
                massB += oriValues[i];
            }

            int value = values[i];
            if(removed & value > 0) { //not removed yet but to remove
                R.mass -= value;
                values[i] = 0; //remove entry
            }
        }

        for(int mode=0; mode<dimension; mode++){
            int[] modeIndices = attributes[mode];
            int[] modeDegree = attValMasses[mode];
            for(int i=0; i<bufferUsage; i++){
                int value = values[i];
                modeDegree[modeIndices[i]] += value;
            }
        }

        if(R.diskUsage > 0) {

            ObjectInputStream inAtt = new ObjectInputStream(new BufferedInputStream(new FileInputStream(R.attFilePath), 8388608));
            ObjectInputStream inOriValue = new ObjectInputStream(new BufferedInputStream(new FileInputStream(Rori.valueFilePath), 8388608));
            ObjectInputStream inCurValue = new ObjectInputStream(new BufferedInputStream(new FileInputStream(R.valueFilePath), 8388608));
            ObjectOutputStream outValue = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream(getFullPath(tempValueName)), 8388608));

            int[] tupleAttValues = new int[dimension];
            while(true) {
                int oriValue = inOriValue.readInt();
                if(oriValue == -1){
                    outValue.writeInt(-1);
                    break;
                }
                boolean removed = true;
                for(int mode = 0; mode < dimension; mode++) {
                    tupleAttValues[mode] = inAtt.readInt();
                    if(!modeToindicesToRemoveArr[mode][tupleAttValues[mode]]) {
                        removed = false;
                    }
                }

                if(removed) {
                    massB += oriValue;
                }

                int value = inCurValue.readInt();
                if(removed & value > 0) { //not removed yet but to remove
                    R.mass -= value;
                    outValue.writeInt(0); //remove entry
                }
                else if (value > 0){ //to remain
                    for (int mode = 0; mode < dimension; mode++) {
                        attValMasses[mode][tupleAttValues[mode]] += value;
                    }
                    outValue.writeInt(value);
                }
                else {
                    outValue.writeInt(value);
                }

            }
            inAtt.close();
            inOriValue.close();
            inCurValue.close();
            outValue.close();
        }

        System.out.println("Block: " + (blockIndex+1));
        System.out.print("Volume: ");
        for(int mode = 0; mode < dimension; mode++) {
            System.out.print(cardinalitiesOfBlock[mode]);
            if(mode < dimension - 1) {
                System.out.print(" X ");
            }
        }
        System.out.println();
        double density = measure.density(massB, cardinalitiesOfBlock);
        System.out.println("Density: " + density);
        System.out.println("Mass: " + massB);

        new File(R.valueFilePath).delete();
        new File(getFullPath(tempValueName)).renameTo(new File(R.valueFilePath));

        return density;
    }

    public void run(final int blockNum, DensityMeasure densityMeasure, final int policy) throws IOException {

        long start = System.currentTimeMillis();
        copyOriTesnor();
        initialize();

        IDensityMeasure measure = null;
        if(densityMeasure == DensityMeasure.Suspiciousness)
            measure = new Suspiciousness();
        else if(densityMeasure == DensityMeasure.Arithmetic)
            measure = new Arithmetic();
        else if(densityMeasure == DensityMeasure.Geometric)
            measure = new Geometric();
        else {
            System.out.println("Error: Unknown Density IMeasure");
        }
        measure.initialize(getDimension(), getCardinalities(), getMass(TensorType.OriginalR));

        final List<BlockInfo> listOfBlocks = new LinkedList();
        double bestAccuracy = 0;
        for(int i = 0; i < blockNum; i++) {
            BlockInfo block = findOneBlock(i, densityMeasure, policy);
            bestAccuracy = Math.max(bestAccuracy, removeAndEvaluateBlock(i, block, measure));
            listOfBlocks.add(block);
        }
        System.out.println("Running time: " + (System.currentTimeMillis() - start + 0.0)/1000 + " seconds");


        start = System.currentTimeMillis();
        System.out.println("Writing outputs...");
        writeOutput(outputPath, Rori, listOfBlocks);
        System.out.println("Outputs were written. " + (System.currentTimeMillis() - start + 0.0) / 1000 + " seconds was taken.");

        System.out.println("Removing temporary files...");
        remove(blockNum);
        System.out.println("Temporary files were removed.");

        return;
    }

    /**
     * Remove Temp Files
     */
    private void remove(int blockNum){
        List<String> filesToRemove = new LinkedList<String>();
        filesToRemove.add(getFullPath(originalAttName));
        filesToRemove.add(getFullPath(originalValueName));
        filesToRemove.add(getFullPath(currentValueName));
        filesToRemove.add(getFullPath(blockAttName));
        filesToRemove.add(getFullPath(blockValueName));

        for(int blockIndex = 0; blockIndex < blockNum; blockIndex++) {
            filesToRemove.add(getBlockInfoFullPath(blockIndex));
            filesToRemove.add(getOrderingFullPath(blockIndex));
        }

        for(String file : filesToRemove) {
            if (new File(file).exists()) {
                new File(file).delete();
            }
        }
    }

    /**
     * find one block from a given tensor
     * @return mode -> list of attributes contained in the block
     */
    private BlockInfo findOneBlock(int blockIndex, DensityMeasure densityMeasure, final int policy) throws IOException {

        final int dimension = getDimension();
        final int[] cardinalities = getCardinalities();
        long mass =  getMass(TensorType.CurrentR);

        // n -> list of attribute values in the nth mode
        final int[][] modeToAttVals = createModeToAttVals(dimension, cardinalities);
        // (n, i) -> mass of ith attribute value in the nth mode
        int[][] modeToAttValToMass = getAttValMasses();
        // n -> num of alive attribute values in the nth mode
        int[] modeToAliveValuesNum = cardinalities.clone();
        // n -> num of deleted attributes in the nth mode
        int[] modeToRemovedValuesNum = new int[dimension];

        final int sumOfcardinalities = sumOfCarndinalities(dimension, cardinalities);
        IDensityMeasure measure = null;
        if(densityMeasure == DensityMeasure.Suspiciousness)
            measure = new Suspiciousness();
        else if(densityMeasure == DensityMeasure.Arithmetic)
            measure = new Arithmetic();
        else if(densityMeasure == DensityMeasure.Geometric)
            measure = new Geometric();
        else {
            System.out.println("Error: Unknown Density IMeasure");
        }

        copyBlock();

        System.gc();

        BlockIterInfo iterInfo = new BlockIterInfo(cardinalities, BlockIterInfo.properBufferUsage(sumOfcardinalities), getOrderingFullPath(blockIndex));
        int maxIters = 0;
        double maxScoreAmongIters = measure.initialize(getDimension(), getCardinalities(), getMass(TensorType.CurrentR));

        int i = 0;
        int iterNum = 0;
        while (i < sumOfcardinalities) {
            int maxMode = 0;
            double maxScoreAmongModes = -Double.MAX_VALUE;
            for (int mode = 0; mode < dimension; mode++) {
                if(modeToAliveValuesNum[mode] > 0) {
                    if(policy == POLICY_MAX_CARNDILITY) {
                        int tempScore = modeToAliveValuesNum[mode];
                        if (tempScore >= maxScoreAmongModes) {
                            maxMode = mode;
                            maxScoreAmongModes = tempScore;
                        }
                    }
                    else if (policy == POLICY_MAX_DENSITY) {
                        double threshold = mass * 1.0 / modeToAliveValuesNum[mode];
                        int numToRemove = 0;
                        long removedMassSum = 0;
                        int[] attValToMass = modeToAttValToMass[mode];
                        int[] attVals = modeToAttVals[mode];
                        for (int j = modeToRemovedValuesNum[mode]; j < cardinalities[mode]; j++) {
                            int attVal = attVals[j];
                            if (attValToMass[attVal] <= threshold) {
                                numToRemove++;
                                removedMassSum += attValToMass[attVal];
                            }
                        }
                        if (numToRemove >= 1) {
                            double tempScore = measure.ifRemoved(mode, numToRemove, removedMassSum);
                            if (tempScore >= maxScoreAmongModes) {
                                maxMode = mode;
                                maxScoreAmongModes = tempScore;
                            }
                        } else {
                            System.out.println("Sanity Check!");
                        }
                    }
                    else {
                        System.out.println("ERROR"); //unknown mrunning mode
                    }
                }
            }

            double threshold = mass * 1.0 / modeToAliveValuesNum[maxMode];
            final int[] attValToMass = modeToAttValToMass[maxMode];
            final boolean[] attValsToRemove = new boolean[cardinalities[maxMode]];

            sort(modeToAttVals[maxMode], modeToAttValToMass[maxMode], modeToRemovedValuesNum[maxMode], cardinalities[maxMode]-1);

            int[] attVals = modeToAttVals[maxMode];

            for (int j = modeToRemovedValuesNum[maxMode]; j < cardinalities[maxMode]; j++) {
                int attVal = attVals[j];
                if (attValToMass[attVal] <= threshold) {
                    mass -= attValToMass[attVal];
                    double score = measure.remove(maxMode, attValToMass[attVal]);
                    if (score > maxScoreAmongIters) {
                        maxScoreAmongIters = score;
                        maxIters = i + 1;
                    }
                    modeToRemovedValuesNum[maxMode]++;
                    modeToAliveValuesNum[maxMode]--;
                    iterInfo.addIterInfo((byte)maxMode, attVal);
                    i++;
                    attValsToRemove[attVal] = true;
                }
                else {
                    break;
                }
            }
            removeAndUpdateAttValMasses(maxMode, attValsToRemove, modeToAttValToMass, iterNum == 0);
            iterNum ++;

        }

        //free attValMasses info
        for(int mode = 0; mode < dimension; mode++) {
            attValMasses[mode] = new int[0];
        }
        return iterInfo.returnBlock(maxIters, getBlockInfoFullPath(blockIndex));
    }

    private static int sumOfCarndinalities(int dimension, int[] cardinalities){
        int sumOfcardinalities = 0;
        for(int mode = 0; mode < dimension; mode++) {
            sumOfcardinalities += cardinalities[mode];
        }
        return sumOfcardinalities;
    }

    /**
     *
     * @param dimension
     * @param cardinalities
     * @return
     */
    private static int[][] createModeToAttVals(final int dimension, int[] cardinalities) {
        int[][] modeToIndices = new int[dimension][];
        for(int mode = 0; mode < dimension; mode++) {
            int[] indices = new int[cardinalities[mode]];
            for(int index = 0; index < cardinalities[mode]; index++) {
                indices[index] = index;
            }
            modeToIndices[mode] = indices;
        }
        return modeToIndices;
    }

    /**
     * sort
     * @param attributes
     * @param masses
     * @param left
     * @param right
     */
    private static void sort(int[] attributes, int[] masses, int left, int right) {

        if (attributes == null || attributes.length == 0)
            return;

        if (left >= right)
            return;

        int middle = left + (right - left) / 2;
        int pivot = masses[attributes[middle]];

        int i = left, j = right;
        while (i <= j) {
            while (masses[attributes[i]] < pivot) {
                i++;
            }

            while (masses[attributes[j]] > pivot) {
                j--;
            }

            if (i <= j) {
                int temp = attributes[i];
                attributes[i] = attributes[j];
                attributes[j] = temp;
                i++;
                j--;
            }
        }

        if (left < j)
            sort(attributes, masses, left, j);

        if (right > i)
            sort(attributes, masses, i, right);
    }

    /**
     * write blocks found to the given output folder
     * @param output    output path
     * @param tensor    tensor
     * @param blockInfoList   blocks found
     * @throws IOException
     */
    private static void writeOutput(String output, Tensor tensor, List<BlockInfo> blockInfoList) throws IOException {

        int blockNum = blockInfoList.size();
        int dimension = tensor.dimension;
        for(int blockIndex = 0; blockIndex < blockNum; blockIndex++) {

            //write attribute values
            BufferedWriter bw = new BufferedWriter(new FileWriter(output + File.separator + "block_"+(blockIndex+1)+".attributes"));
            final boolean[][] attributeToValuesToWrite = new boolean[tensor.dimension][];
            BlockInfo block = blockInfoList.get(blockIndex);
            Set<Integer>[] attributeToValues = block.getAttributeValues(dimension);
            for(int dim = 0; dim < tensor.dimension; dim++) {
                attributeToValuesToWrite[dim] = new boolean[tensor.cardinalities[dim]];
                for(int value : attributeToValues[dim]) {
                    attributeToValuesToWrite[dim][value] = true;
                    bw.write(dim+","+value);
                    bw.newLine();
                }
            }
            bw.close();

            //write blocks
            bw = new BufferedWriter(new FileWriter(output + File.separator + "block_"+(blockIndex+1)+".tuples"));
            final int[][] attributes = tensor.attributes;
            final int[] measureValues = tensor.measureValues;
            for(int i=0; i<tensor.bufferUsage; i++) {
                boolean write = true;
                for(int dim = 0; dim < dimension; dim++) {
                    if(!attributeToValuesToWrite[dim][attributes[dim][i]]) {
                        write = false;
                        break;
                    }
                }
                if(write) {
                    for(int dim = 0; dim < dimension; dim++) {
                        bw.write(attributes[dim][i] + ",");
                    }
                    bw.write(""+measureValues[i]);
                    bw.newLine();
                }
            }

            if(tensor.diskUsage > 0) {
                ObjectInputStream inAtt = new ObjectInputStream(new BufferedInputStream(new FileInputStream(tensor.attFilePath), 8388608));
                ObjectInputStream inValue = new ObjectInputStream(new BufferedInputStream(new FileInputStream(tensor.valueFilePath), 8388608));

                int[] tupleAttValues = new int[dimension];
                while (true) {
                    int value = inValue.readInt();
                    if (value == -1) {
                        break;
                    }
                    for (int dim = 0; dim < dimension; dim++) {
                        tupleAttValues[dim] = inAtt.readInt();
                    }
                    boolean write = true;
                    for (int dim = 0; dim < dimension; dim++) {
                        if (!attributeToValuesToWrite[dim][tupleAttValues[dim]]) {
                            write = false;
                            break;
                        }
                    }
                    if (write) {
                        for (int dim = 0; dim < dimension; dim++) {
                            bw.write(tupleAttValues[dim] + ",");
                        }
                        bw.write("" + value);
                        bw.newLine();
                    }
                }
                inAtt.close();
                inValue.close();
            }
            bw.close();
        }
    }
}