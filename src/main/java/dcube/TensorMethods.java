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
 * Methods for handling tensors
 * @author kijungs
 */
public class TensorMethods {
	

    public static Tensor importSparseTensor(final String path, final String delim, final int dimension, final int[] modeLengths, final int bufferSize, String attFilePath, String valueFilePath) throws IOException {

        long start = System.currentTimeMillis();

        final int[][] attVals = new int[dimension][bufferSize];
        final int[] values = new int[bufferSize];

        final BufferedReader br = new BufferedReader(new FileReader(path));
        long sum = 0;
        long omega = 0;
        for(int i=0; i<bufferSize; i++) {
            String line = br.readLine();
            String[] tokens = line.split(delim);
            if(tokens.length < dimension + 1) {
                System.out.println("Skipped Line: " + line);
                continue;
            }
            for (int mode = 0; mode < dimension; mode++) {
                int index = Integer.valueOf(tokens[mode]);
                attVals[mode][i] = index;
            }
            values[i] = Integer.valueOf(tokens[dimension]);
            sum += values[i];
            omega++;
        }

        ObjectOutputStream outAtt = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream(attFilePath), 8388608));
        ObjectOutputStream outValue = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream(valueFilePath), 8388608));
        while(true) {
            String line = br.readLine();
            if(line == null) {
                outValue.writeInt(-1);
                break;
            }
            String[] tokens = line.split(delim);
            if(dimension == tokens.length + 1){
                System.out.println("Error. The following line is ignored");
                System.out.println(line);
                continue;
            }
            int value = Integer.valueOf(tokens[dimension]);
            sum += value;
            omega ++;
            outValue.writeInt(value);
            for(int mode = 0; mode<dimension; mode++) {
                outAtt.writeInt(Integer.valueOf(tokens[mode]));
            }
        }

        br.close();
        outAtt.close();
        outValue.close();

        System.out.println("Preprocess," + (System.currentTimeMillis() - start));

        return new Tensor(dimension, modeLengths, attVals, values, omega, sum, bufferSize, bufferSize, attFilePath, valueFilePath);
    }

    /**
     * compute the weighted attValMasses
     * @param tensor
     * @return (mode index) -> weighted attValMasses
     */
    public static int[][] attributeValueMasses(Tensor tensor) throws IOException {

        int dimension = tensor.dimension;
        int[] modeLengths = tensor.cardinalities;
        int[][] attValMasses = new int[dimension][];
        for(int mode=0; mode<dimension; mode++){
            attValMasses[mode] = new int[modeLengths[mode]];
        }

        int[][] attVals = tensor.attributes;
        int[] values = tensor.measureValues;

        for(int mode=0; mode<dimension; mode++){
            int[] modeAttVals = attVals[mode];
            int[] modeAttValMasses = attValMasses[mode];
            for(int i=0; i<tensor.bufferUsage; i++){
                int value = values[i];
                modeAttValMasses[modeAttVals[i]] += value;
            }
        }

        if(tensor.diskUsage > 0 ){
            ObjectInputStream inAtt = new ObjectInputStream(new BufferedInputStream(new FileInputStream(tensor.attFilePath), 8388608));
            ObjectInputStream inValue = new ObjectInputStream(new BufferedInputStream(new FileInputStream(tensor.valueFilePath), 8388608));
            while (true) {
                int value = inValue.readInt();
                if (value == -1) {
                    break;
                }
                for (int mode = 0; mode < dimension; mode++) {
                    attValMasses[mode][inAtt.readInt()] += value;
                }
            }
            inAtt.close();
            inValue.close();
        }

        return attValMasses;
    }

}
