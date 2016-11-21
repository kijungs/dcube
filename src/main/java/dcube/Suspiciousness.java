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
 * Suspiciousness, one of density measures
 * @author kijungs
 */
public class Suspiciousness implements IDensityMeasure {

    private int dimension;
    private int[] cardinalities;
    private long massOfAll; //sum of entries
    private long massOfBlock;
    private double productOfCardinalitiesOfAll;
    private double productOfCardinalitiesOfBlock;

    public double initialize(int dimension, int[] cardinalities, long mass) {
        this.dimension = dimension;
        this.cardinalities = cardinalities.clone();
        this.massOfAll = mass;
        this.massOfBlock = mass;
        productOfCardinalitiesOfAll = 1;
        for(int dim = 0; dim < dimension; dim++) {
            productOfCardinalitiesOfAll *= cardinalities[dim];
        }
        productOfCardinalitiesOfBlock = productOfCardinalitiesOfAll;
        return density(massOfBlock, productOfCardinalitiesOfBlock);
    }

    public double ifRemoved(int attribute, int mass) {
        return density(massOfBlock - mass, productOfCardinalitiesOfBlock / cardinalities[attribute] * (cardinalities[attribute] - 1));
    }

    public double ifRemoved(int attribute, int numValues, long sumOfMasses) {
        return density(massOfBlock - sumOfMasses, productOfCardinalitiesOfBlock / cardinalities[attribute] * (cardinalities[attribute] - numValues));
    }

    public double remove(int attribute, int mass) {
        cardinalities[attribute]--;
        productOfCardinalitiesOfBlock = productOfCardinalities(cardinalities); //recompute due to the precision error
        massOfBlock -= mass;
        return density(massOfBlock, productOfCardinalitiesOfBlock);
    }

    public double density(long massOfBlock, int[] cardinalitiesOfBlock) {
        double productOfCardinalitiesOfBlock = 1;
        for(int dim = 0; dim < dimension; dim++) {
            productOfCardinalitiesOfBlock *= cardinalitiesOfBlock[dim];
        }
        return density(massOfBlock, productOfCardinalitiesOfBlock);
    }

    private double density(long massOfBlock, double productOfCardinalitiesOfBlock) {
        if(productOfCardinalitiesOfBlock == 0 || massOfBlock == 0)
            return - 1;
        return massOfBlock * (Math.log((massOfBlock+0.0)/ massOfAll) - 1) + massOfAll * productOfCardinalitiesOfBlock / productOfCardinalitiesOfAll - massOfBlock * Math.log (productOfCardinalitiesOfBlock / productOfCardinalitiesOfAll);
    }

    public static double productOfCardinalities(int[] cardinalities){
        double productOfCardinalities = 1;
        for(int attribute = 0; attribute < cardinalities.length; attribute++) {
            productOfCardinalities *= cardinalities[attribute];
        }
        return productOfCardinalities;
    }
}
