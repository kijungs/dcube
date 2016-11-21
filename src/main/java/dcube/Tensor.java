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

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;

/**
 * Data structure to store tensor data
 * @author kijungs
 */
public class Tensor {

    public int dimension; //number of modes
    public int[] cardinalities; // n -> cardinality of the n-th attribute
    public int[][] attributes; // (n, i) -> the n-th attribute value of the i-th tuple
    public int[] measureValues; // i -> measure attribute value of i-th tuple
    public long mass; //  sum of measures attributes measureValues
    public int bufferSize; // maximum number of tuples in memory buffer
    public String attFilePath; // path of the attribute file to spill data
    public String valueFilePath; // path of the measure value file to spill data
    public int bufferUsage; // number of entries in the current buffer
    public long diskUsage; // number of entries in the disk

    /**
     *
     * @param dimension	//number of modes
     * @param cardinalities	// n -> cardinality of the n-th attribute
     * @param omega	// number of tuples
     * @param bufferSize // maximum number of tuples in memory buffer
     * @param attFilePath // path of the attribute file to spill data
     * @param valueFilePath // path of the measure value file to spill data
     */
    public Tensor(int dimension, int[] cardinalities, long omega, long mass, int bufferSize, String attFilePath, String valueFilePath) {
        this.dimension = dimension;
        this.cardinalities = cardinalities;
        this.mass = mass;
        this.bufferSize = bufferSize;
        this.attributes = new int[dimension][];
        for(int mode = 0; mode< dimension; mode++) {
            this.attributes[mode] = new int[bufferSize];
        }
        this.measureValues = new int[bufferSize];
        this.attFilePath = attFilePath;
        this.valueFilePath = valueFilePath;
        this.bufferUsage = 0;
        this.diskUsage = omega;
    }

    public Tensor(int dimension, int[] cardinalities, int[][] attributes, int[] values, long omega, long mass, int bufferSize, int bufferUsage, String attFilePath, String valueFilePath) {
        this.dimension = dimension;
        this.cardinalities = cardinalities;
        this.mass = mass;
        this.bufferSize = bufferSize;
        this.attributes = attributes;
        this.measureValues = values;
        this.attFilePath = attFilePath;
        this.valueFilePath = valueFilePath;
        this.bufferUsage = bufferUsage;
        this.diskUsage = omega - bufferUsage;
    }


    public Tensor(Tensor tensor, String valueFilePath) throws IOException {
        this.dimension = tensor.dimension;
        this.cardinalities = tensor.cardinalities;
        this.mass = tensor.mass;
        this.bufferSize = tensor.bufferSize;
        this.attributes = tensor.attributes;
        this.measureValues = tensor.measureValues.clone(); //do not share measureValues
        this.attFilePath = tensor.attFilePath;
        this.valueFilePath = valueFilePath;  //do not share measureValues
        if(tensor.diskUsage > 0) {
            Files.copy(new File(tensor.valueFilePath).toPath(), new File(valueFilePath).toPath(), StandardCopyOption.REPLACE_EXISTING);
        }
        this.bufferUsage = tensor.bufferUsage;
        this.diskUsage = tensor.diskUsage;
    }

    public Tensor copy(String valueFilePath) throws IOException {
        return new Tensor(this, valueFilePath);
    }
}
