package return_of_gradient_descent;

/**
 * The HingeLoss loss function for classification <i>L(x, y) = max(0, 1-y*x)</i>.
 * <br>
 * This function is only once differentiable.
 *
 * @author Edward Raff
 */
public class HingeLoss
{

    /**
     * Computes the HingeLoss loss
     *
     * @param pred the predicted value
     * @param y the target value
     * @return the HingeLoss loss
     */
    public static double loss(double pred, double y)
    {
        return Math.max(0, 1 - y * pred);
    }

    /**
     * Computes the first derivative of the HingeLoss loss
     *
     * @param pred the predicted value
     * @param y the target value
     * @return the first derivative of the HingeLoss loss
     */
    public static double deriv(double pred, double y)
    {
        if (pred * y > 1)
            return 0;
        else
            return -y;
    }

}
