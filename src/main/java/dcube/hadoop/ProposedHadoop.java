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

package dcube.hadoop;

import dcube.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.*;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.Set;

/**
 * D-Cube Hadoop Version
 * @author kijungs
 */
public class ProposedHadoop {


    /**
     * Main function
     *
     * @param args input_path, output_path, num_of_attributes, density_measure, num_of_blocks
     * @throws IOException
     */
    public static void main(String[] args) throws Exception {
        if (args.length < 5) {
            printError();
            System.exit(-1);
        }

        final String input = args[0];
        System.out.println("input_path: " + input);

        final String output = args[1];
        System.out.println("output_path: " + output);

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
            policy = Proposed.POLICY_MAX_CARNDILITY;
        } else if(args[4].compareToIgnoreCase("DENSITY") == 0) {
            policy = Proposed.POLICY_MAX_DENSITY;
        } else {
            System.err.println("Unknown Policy");
            printError();
            System.exit(-1);
        }
        System.out.println("policy: " + args[4]);

        final int blockNum = Integer.valueOf(args[5]);
        System.out.println("num_of_blocks: " + blockNum);

        int reducerNum = Integer.valueOf(args[6]);
        System.out.println("num_of_reducers: " + reducerNum);

        String logPath = args[7];
        System.out.println("log path (local): " + logPath);
        File dir = new File(logPath);

        try{
            dir.mkdir();
        }
        catch(Exception e){
        }

        try {
            System.setOut(new PrintStream(new File(logPath+"/log.txt")));
        } catch (Exception e) {
            e.printStackTrace();
        }

        System.out.println("input path: " + input);
        System.out.println("output path: " + output);
        System.out.println("dimension: " + dimension);
        System.out.println("densityMeasure: " +  args[3]);
        System.out.println("policy: " +  args[4]);
        System.out.println("num_of_blocks: " + blockNum);
        System.out.println("reducerNum: " + reducerNum);
        System.out.println("log path (local): " + logPath);

        System.out.println();
        System.out.println("running the algorithm...");
        ProposedHadoop proposed = new ProposedHadoop(input, output, logPath, dimension, reducerNum);
        proposed.run(blockNum, densityMeasure, policy);

    }

    private static void printError() {
        System.err.println("Usage: run_hadoop.sh input_path output_path dimension density_measure policy num_of_blocks num_of_reducers log_path");
        System.err.println("Density_measure should be one of [ari, geo, susp]");
        System.err.println("Policy should be one of [density, cardinality]");
    }

    protected enum TensorType{
        OriginalR, CurrentR
    }

    private String originalPath;
    private String outputPath;
    private String logPath;
    private String defaultCurrentRPath;
    private String defaultBlockBPath;
    private String tempPath;
    private String distributedCachePath;

    //changeable
    public String currentRPath;
    public String blockBPath;

    private int dimension;
    private int reducerNum;
    private int[] cardinalities;
    private int[][] attValMasses;
    private long massR;
    private long massRori;

    private Configuration conf;

    public void run(final int blockNum, DensityMeasure densityMeasure, final int runningMode) throws Exception {

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
            BlockInfo block = findOneBlock(i, densityMeasure, runningMode);
            bestAccuracy = Math.max(bestAccuracy, removeAndEvaluateBlock(i, block, measure, i == blockNum -1));
            listOfBlocks.add(block);
        }

        System.out.println("Running time: " + (System.currentTimeMillis() - start + 0.0)/1000 + " seconds");


        start = System.currentTimeMillis();
        System.out.println("Writing outputs...");
        writeOutput(outputPath, listOfBlocks);
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

    private String getTempLocalFilePath() {
        while (true) {
            String name = "DECUBE_" + new Random().nextInt();
            if(!new File(name).exists())
                return name;
        }
    }

    /**
     * find one block from a given tensor
     * @return # of ieterations when the density is maximized
     */
    private BlockInfo findOneBlock(int blockIndex, DensityMeasure densityMeasure, final int policy) throws Exception {

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

        final int sumOfCardinalities = sumOfCarndinalities(dimension, cardinalities);
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

        BlockIterInfo iterInfo = new BlockIterInfo(cardinalities, BlockIterInfo.properBufferUsage(sumOfCardinalities), getOrderingFullPath(blockIndex));
        int maxIters = 0;
        double maxScoreAmongIters = measure.initialize(getDimension(), getCardinalities(), getMass(TensorType.CurrentR));


        int i = 0;
        while (i < sumOfCardinalities) {
            int maxMode = 0;
            double maxScoreAmongModes = -Double.MAX_VALUE;
            for (int mode = 0; mode < dimension; mode++) {
                if(modeToAliveValuesNum[mode] > 0) {
                    if(policy == Proposed.POLICY_MAX_CARNDILITY) {
                        int tempScore = modeToAliveValuesNum[mode];
                        if (tempScore >= maxScoreAmongModes) {
                            maxMode = mode;
                            maxScoreAmongModes = tempScore;
                        }
                    }
                    else if (policy == Proposed.POLICY_MAX_DENSITY) {
                        double threshold = mass * (1.0) / modeToAliveValuesNum[mode];
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

            removeAndUpdateAttValMasses(maxMode, attValsToRemove, modeToAttValToMass);
        }

        //free attValMasses info
        for(int mode = 0; mode < dimension; mode++) {
            attValMasses[mode] = new int[0];
        }
        return iterInfo.returnBlock(maxIters, getBlockInfoFullPath(blockIndex));
    }

    private static int sumOfCarndinalities(int order, int[] modeLengths){
        int sumOfModeLengths = 0;
        for(int mode = 0; mode < order; mode++) {
            sumOfModeLengths += modeLengths[mode];
        }
        return sumOfModeLengths;
    }

    /**
     * @return
     */
    private static int[][] createModeToAttVals(final int order, int[] modeLengths) {
        int[][] modeToAttVals = new int[order][];
        for(int mode = 0; mode < order; mode++) {
            int[] attVals = new int[modeLengths[mode]];
            for(int attVal = 0; attVal < modeLengths[mode]; attVal++) {
                attVals[attVal] = attVal;
            }
            modeToAttVals[mode] = attVals;
        }
        return modeToAttVals;
    }

    public static void sort(int[] indices, int[] masses) {
        sort(indices, masses, 0, indices.length-1);
    }
    public static void sort(int[] indices, int[] masses, int left, int right) {

        if (indices == null || indices.length == 0)
            return;

        if (left >= right)
            return;

        int middle = left + (right - left) / 2;
        int pivot = masses[indices[middle]];

        int i = left, j = right;
        while (i <= j) {
            while (masses[indices[i]] < pivot) {
                i++;
            }

            while (masses[indices[j]] > pivot) {
                j--;
            }

            if (i <= j) {
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
                i++;
                j--;
            }
        }

        if (left < j)
            sort(indices, masses, left, j);

        if (right > i)
            sort(indices, masses, i, right);
    }

    public ProposedHadoop(String inputPath, String outputPath, String logPath, int dimension, int reducerNum) throws Exception {

        this.originalPath = inputPath;
        this.outputPath = outputPath;
        this.logPath = logPath;
        this.dimension = dimension;
        cardinalities = new int[dimension];
        attValMasses = new int[dimension][];
        this.reducerNum = reducerNum;

        defaultCurrentRPath = outputPath + "/disk_cur";
        defaultBlockBPath = outputPath + "/disk_block";
        tempPath = outputPath + "/temp";
        currentRPath = outputPath + "/disk_cur";
        blockBPath = outputPath + "/disk_block";
        distributedCachePath = outputPath +"/cache";

        conf = new Configuration();
        conf.setInt(Parameter.PARAM_DIMENSION, dimension);
        conf.setInt(Parameter.PARAM_REDUCER_NUM, reducerNum);

        long fileSize = getFileSize(originalPath);
        conf.setBoolean("mapred.map.tasks.speculative.execution", false);
        conf.setInt("mapred.map.tasks", reducerNum);
        conf.setLong("mapred.min.split.size", ((fileSize/reducerNum)+1L));
        conf.setLong("mapred.max.split.size", ((fileSize/reducerNum)+1L));

    }

    private long getFileSize(String filePath) throws IOException {

        long size = 0;
        FileSystem fs = FileSystem.get(conf);
        FileStatus[] statusList = fs.listStatus(new Path(filePath));
        for(FileStatus status : statusList){
            size += status.getLen();
        }
        return size;
    }


    public void initialize() throws Exception {
        runCardinalityJob();
        readCardinalities();
        runAttValMassJob(originalPath);
        readAttValMasses(attValMasses);
        massR = computeMass(attValMasses);
        massRori = massR;
    }

    private void runCardinalityJob() throws Exception {
        FileSystem fs = FileSystem.get(conf);
        if(fs.exists(new Path(tempPath))) {
            fs.delete(new Path(tempPath), true);
        }

        Job job = new Job(conf, "ModeLength");

        job.setJarByClass(CardinalityMR.class);
        job.setMapperClass(CardinalityMR.CardinalityMapper.class);

        job.setNumReduceTasks(1);
        job.setReducerClass(CardinalityMR.CardinalityReducer.class);
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(IntWritable.class);
        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(originalPath));
        FileOutputFormat.setOutputPath(job, new Path(tempPath));

        job.waitForCompletion(true);
    }

    private void readCardinalities() throws  Exception {
        FileSystem fs = FileSystem.get(conf);
        String parentPath = tempPath;
        FileStatus[] statusList = fs.listStatus(new Path(parentPath));

        for(FileStatus status : statusList){
            String fileName = status.getPath().getName();
            if (fileName.contains("part")) {
                BufferedReader in = new BufferedReader(new InputStreamReader(fs.open(new Path(parentPath + "/" + fileName))));
                while (true) {
                    String line = in.readLine();
                    if(line==null)
                        break;
                    String[] tokens = line.split(",");
                    int mode = Integer.valueOf(tokens[0]);
                    int modeLength = Integer.valueOf(tokens[1]);
                    cardinalities[mode] = modeLength;
                }
                in.close();
            }
        }

        for(int mode = 0; mode< dimension; mode++){
            conf.setInt(Parameter.PARAM_CARDINALITY + mode, cardinalities[mode]);
            System.out.println("mode lengh (mode=" + mode + "): " + cardinalities[mode]);
        }
    }

    protected long getMass(TensorType type) {
        if(type == TensorType.OriginalR) {
            return massRori;
        }
        else {
            return massR;
        }
    }

    protected int[] getCardinalities() {
        return cardinalities;
    }

    protected int[][] getAttValMasses() {
        return attValMasses;
    }

    protected int getDimension() {
        return dimension;
    }

    protected void copyOriTesnor() throws IOException {
        currentRPath = originalPath;
    }

    protected void copyBlock() throws IOException, ClassNotFoundException, InterruptedException {
        blockBPath = currentRPath;
    }

    protected void removeAndUpdateAttValMasses(int mode, boolean[] attValsToRemove, int[][] attValMasses) throws Exception {
        runRemoveBJob(mode, attValsToRemove);
        runAttValMassJob(blockBPath);
        readAttValMasses(attValMasses);
    }

    public void runRemoveBJob(int mode, boolean[] attValsToRemove) throws Exception {

        conf.setInt(Parameter.PARAM_MODE_TO_REMOVE, mode);

        //create a file to distribute
        String fileToDistribute = getTempLocalFilePath();
        ObjectOutputStream out = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream(fileToDistribute), 8388608));
        for(int att=0; att<attValsToRemove.length; att++) {
            if(attValsToRemove[att]) {
                out.writeInt(att);
            }
        }
        out.close();

        //upload a file to distribute
        FileSystem fs = FileSystem.get(conf);
        fs.copyFromLocalFile(true, true, new Path(fileToDistribute), new Path(distributedCachePath));

        if(fs.exists(new Path(tempPath))) {
            fs.delete(new Path(tempPath), true);
        }

        Job job = new Job(conf, "RemoveB");
        job.setJarByClass(RemoveBMR.class);
        job.setMapperClass(RemoveBMR.RemoveBMapper.class);

        job.setNumReduceTasks(0); //map only job
        job.setReducerClass(RemoveBMR.RemoveBReducer.class);
        job.setMapOutputKeyClass(NullWritable.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(blockBPath));
        FileOutputFormat.setOutputPath(job, new Path(tempPath));

        DistributedCache.addCacheFile(new Path(distributedCachePath).toUri(), job.getConfiguration());

        job.waitForCompletion(true);

        blockBPath = defaultBlockBPath; //blockBPath can be set to currentRPath
        if(fs.exists(new Path(blockBPath))) {
            fs.delete(new Path(blockBPath), true);
        }
        fs.rename(new Path(tempPath), new Path(blockBPath));

    }

    public void runAttValMassJob(String inputPath) throws Exception {

        FileSystem fs = FileSystem.get(conf);
        if(fs.exists(new Path(tempPath))) {
            fs.delete(new Path(tempPath), true);
        }

        Job job = new Job(conf, "AttValMass");
        job.setJarByClass(AttValMassMR.class);
        job.setMapperClass(AttValMassMR.AttValMassMapper.class);

        job.setNumReduceTasks(reducerNum);
        job.setReducerClass(AttValMassMR.AttValMassReducer.class);
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(IntWritable.class);
        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(inputPath));
        FileOutputFormat.setOutputPath(job, new Path(tempPath));

        job.waitForCompletion(true);
    }

    public void readAttValMasses(int[][] attValMasses) throws IOException {

        FileSystem fs = FileSystem.get(conf);
        String parentPath = tempPath;
        FileStatus[] statusList = fs.listStatus(new Path(parentPath));
        for(int mode = 0; mode < dimension; mode++) {
            attValMasses[mode] = null;
            attValMasses[mode] = new int[cardinalities[mode]];
        }

        //read attribute-value masses
        for(FileStatus status : statusList){
            String fileName = status.getPath().getName();
            if (fileName.contains("part")) {
                BufferedReader in = new BufferedReader(new InputStreamReader(fs.open(new Path(parentPath + "/" + fileName))));
                while (true) {
                    String line = in.readLine();
                    if (line == null)
                        break;
                    String[] tokens = line.split(",");
                    int mode = Integer.valueOf(tokens[0]);
                    int att = Integer.valueOf(tokens[1]);
                    int sum = Integer.valueOf(tokens[2]);
                    attValMasses[mode][att] = sum;
                }
                in.close();
            }
        }
    }

    protected double removeAndEvaluateBlock(int blockIndex, BlockInfo block, IDensityMeasure measure, final boolean isLastIter) throws Exception {
        runEvaluateJob(block);
        int[] cardinalitiesOfBlock = block.blockCardinalities;
        long massB = readEvaluateResult();

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


        if(isLastIter) {
            return density;
        }

        runRemoveRJob();
        runAttValMassJob(currentRPath);
        readAttValMasses(attValMasses);
        massR = computeMass(attValMasses);

        return density;
    }

    private void runEvaluateJob(BlockInfo block) throws Exception {

        //create a file to distribute
        String fileToDistribute = block.returnFileInfo(getTempLocalFilePath());

        //upload a file to distribute
        FileSystem fs = FileSystem.get(conf);
        fs.copyFromLocalFile(true, true, new Path(fileToDistribute), new Path(distributedCachePath));

        if(fs.exists(new Path(tempPath))) {
            fs.delete(new Path(tempPath), true);
        }

        Job job = new Job(conf, "Evaluate");
        job.setJarByClass(EvaluateMR.class);
        job.setMapperClass(EvaluateMR.EvaluateMapper.class);

        job.setNumReduceTasks(1);
        job.setReducerClass(EvaluateMR.EvaluateReducer.class);
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(LongWritable.class);
        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(originalPath));
        FileOutputFormat.setOutputPath(job, new Path(tempPath));

        DistributedCache.addCacheFile(new Path(distributedCachePath).toUri(), job.getConfiguration());

        job.waitForCompletion(true);
    }


    private long readEvaluateResult() throws IOException {

        FileSystem fs = FileSystem.get(conf);
        String parentPath = tempPath;
        FileStatus[] statusList = fs.listStatus(new Path(parentPath));

        //read mass
        long mass = 0;
        for(FileStatus status : statusList){
            String fileName = status.getPath().getName();
            if (fileName.contains("part")) {
                BufferedReader in = new BufferedReader(new InputStreamReader(fs.open(new Path(parentPath + "/" + fileName))));
                while (true) {
                    String line = in.readLine();
                    if (line == null)
                        break;
                    mass += Long.valueOf(line);
                }
                in.close();
            }
        }

        return mass;
    }

    private void runRemoveRJob() throws Exception {

        FileSystem fs = FileSystem.get(conf);

        if(fs.exists(new Path(tempPath))) {
            fs.delete(new Path(tempPath), true);
        }

        Job job = new Job(conf, "RemoveR");
        job.setJarByClass(RemoveRMR.class);
        job.setMapperClass(RemoveRMR.RemoveRMapper.class);

        job.setNumReduceTasks(0);
        job.setReducerClass(RemoveRMR.RemoveRReducer.class);
        job.setMapOutputKeyClass(NullWritable.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(currentRPath));
        FileOutputFormat.setOutputPath(job, new Path(tempPath));

        DistributedCache.addCacheFile(new Path(distributedCachePath).toUri(), job.getConfiguration()); //reuse

        job.waitForCompletion(true);

        currentRPath = defaultCurrentRPath;
        if(fs.exists(new Path(currentRPath))) {
            fs.delete(new Path(currentRPath), true);
        }
        fs.rename(new Path(tempPath), new Path(currentRPath));

    }

    private long computeMass(int[][] degree) {
        long massR = 0;
        for(int index = 0; index < cardinalities[0]; index++){
            massR += degree[0][index];
        }
        return massR;
    }

    private String getOrderingFullPath(int blockIndex) {
        return logPath + File.separator + "ordering_info" + blockIndex;
    }

    private String getBlockInfoFullPath(int blockIndex) {
        return logPath + File.separator + "block_info" + blockIndex;
    }

    /**
     * write blocks found to the given output folder
     * @param output    output path
     * @param blockInfoList   blocks found
     * @throws IOException
     */
    private void writeOutput(String output, List<BlockInfo> blockInfoList) throws Exception {

        int blockNum = blockInfoList.size();
        for(int blockIndex = 0; blockIndex < blockNum; blockIndex++) {

            //write attribute values
            String attPath = getTempLocalFilePath();
            BufferedWriter bw = new BufferedWriter(new FileWriter(attPath));
            final boolean[][] attributeToValuesToWrite = new boolean[dimension][];
            BlockInfo block = blockInfoList.get(blockIndex);
            Set<Integer>[] attributeToValues = block.getAttributeValues(dimension);
            for (int dim = 0; dim < dimension; dim++) {
                attributeToValuesToWrite[dim] = new boolean[cardinalities[dim]];
                for (int value : attributeToValues[dim]) {
                    attributeToValuesToWrite[dim][value] = true;
                    bw.write(dim + "," + value);
                    bw.newLine();
                }
            }
            bw.close();
            FileSystem fs = FileSystem.get(conf);
            fs.copyFromLocalFile(true, true, new Path(attPath), new Path(output + "/block_" + (blockIndex + 1) + ".attributes"));
            runComputeBOriJob(blockIndex, block);

        }
    }

    private void runComputeBOriJob(int blockIndex, BlockInfo block) throws Exception {

        //create a file to distribute
        String fileToDistribute = block.returnFileInfo(getTempLocalFilePath());

        //upload a file to distribute
        FileSystem fs = FileSystem.get(conf);
        fs.copyFromLocalFile(true, true, new Path(fileToDistribute), new Path(distributedCachePath));

        String blockOutputPath = outputPath +"/block_"+(blockIndex+1)+".tuples";

        Job job = new Job(conf, "ComputeBOri");
        job.setJarByClass(ComputeBOriMR.class);
        job.setMapperClass(ComputeBOriMR.ComputeBOriMapper.class);

        job.setNumReduceTasks(0);
        job.setMapOutputKeyClass(NullWritable.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(originalPath));
        FileOutputFormat.setOutputPath(job, new Path(blockOutputPath));

        DistributedCache.addCacheFile(new Path(distributedCachePath).toUri(), job.getConfiguration());

        job.waitForCompletion(true);

    }
}
