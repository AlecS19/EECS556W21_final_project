## Performance functions



function MCR(test_image, segmented_image)
    """
    INPUTS

    test_image: Image from testing algorithm, with segmented valued pixels
    segmented_image: Image with correctly segmented valued pixels

    OUTPUT

    MRC: outputs a singular MCR number
    MCR = # of mis-classified pixel /  total number of pixels
    """

    m,n = size(test_image)
    total_pixel = m*n

    misclassified = sum( test_image .!= segmented_image )

    return misclassified/total_pixel
end

function PR()
end
