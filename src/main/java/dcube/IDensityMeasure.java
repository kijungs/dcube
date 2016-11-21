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

/**
 * common interface for density measures
 * @author kijungs
 */
public interface IDensityMeasure {

    /**
     * initialize a density measure for a given tensor
     * @param dimension
     * @param cardinalities
     * @param mass
     * @return
     */
    double initialize(int dimension, int[] cardinalities, long mass);

    /**
     * return density if an attribute value with a given mass is removed from a given attribute
     * @param attribute
     * @param mass
     * @return
     */
    double ifRemoved(int attribute, int mass);

    /**
     * return density if the given number of values with the given mass sum are removed from the given attribute
     * @param attribute
     * @param numValues
     * @param sumOfMasses
     * @return
     */
    double ifRemoved(int attribute, int numValues, long sumOfMasses);

    /**
     * return density after removing an attribute value with a given mass from a given attribute
     * @param attribute
     * @param mass
     * @return
     */
    double remove(int attribute, int mass);

    /**
     * return density of a block with a given mass and cardinalities
     * @param mass
     * @param cardinalities
     * @return
     */
    double density(long mass, int[] cardinalities);
}
