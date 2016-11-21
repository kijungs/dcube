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
 * Geometric Average Mass, one of density measures
 * @author kijungs
 */
public class Geometric implements IDensityMeasure {

    private int dimension;
    private int[] cardinalities;
    private long mass;
    private double productOfCardinalities;

    public double initialize(int dimension, int[] cardinalities, long mass) {
        this.dimension = dimension;
        this.cardinalities = cardinalities.clone();
        this.mass = mass;
        productOfCardinalities = 1;
        for(int dim = 0; dim < dimension; dim++) {
            productOfCardinalities *= cardinalities[dim];
        }
        return density(mass, productOfCardinalities);
    }

    public double ifRemoved(int attribute, int mass) {
        return density(this.mass - mass, productOfCardinalities / cardinalities[attribute] * (cardinalities[attribute] - 1));
    }

    public double ifRemoved(int attribute, int numValues, long sumOfMasses) {
        return density(this.mass - sumOfMasses, productOfCardinalities / cardinalities[attribute] * (cardinalities[attribute] - numValues));
    }

    public double remove(int attribute, int mass) {
        cardinalities[attribute]--;
        productOfCardinalities = Suspiciousness.productOfCardinalities(cardinalities); //recompute due to the precision error
        this.mass -= mass;
        return density(this.mass, productOfCardinalities);
    }

    public double density(long mass, int[] cardinalities) {
        double productOfCardinalities = 1;
        for(int dim = 0; dim < dimension; dim++) {
            productOfCardinalities *= cardinalities[dim];
        }
        return density(mass, productOfCardinalities);
    }

    private double density(double mass, double productOfCardinalities) {
        if(productOfCardinalities == 0)
            return - 1;
        return mass / Math.pow(productOfCardinalities, 1.0/dimension);
    }
}
