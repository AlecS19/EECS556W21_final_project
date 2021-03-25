using Clustering: randindex

## Performance functions
function MCR(test_image, segmented_image)
    """
    INPUTS

    test_image: Image from testing algorithm, with segmented valued pixels
    segmented_image: Image with correctly segmented valued pixels

    OUTPUT

    MRC: outputs a singular MCR number
    MRC = # of mis-classified pixel /  total number of pixels
    """

    m,n = size(test_image)
    total_pixel = m*n

    misclassified = sum( test_image .!= segmented_image )

    return misclassified/total_pixel
end

function PR_slow(test_image, ground_truths)
    """
    INPUT
    test_image: 2D Array for the image that is segmented
    segmented_images: 2D Arrat of Dictionart with Segmentation and Boundaries

    #Loop Through the pixel as i
    ##Loop through the pixels j compared to i, i neq j
    ###Calculate p_ij
    ####Loop through all ground truth images at pixel location/ total number of ground truths
    ###add pij if i and j are labelled the same, 1-pij

    OUTPUT
    PR: The PR score of the test image
    """

    s = "Segmentation"
    numGT = size( ground_truths, 2)
    PR = 0

    #flatten the image
    test_image_flat = vec( test_image )
    m,n = size(test_image)
    mn = m*n


    for i in 1:mn
        if (i % 5000 == 0)
            println(i)
            println(PR)
        end

        for j in 1:i-1
            p_ij = 0

            for k in 1:numGT
                segment = vec( ground_truths[k][s] )
                if ( segment[i] == segment[j] )
                    p_ij += 1
                end
            end

            p_ij /= numGT

            if ( test_image_flat[i] == test_image_flat[j] )
                PR += p_ij
            else
                PR += (1 - p_ij)
            end
        end
    end

    return PR
end

function PR_fast(test_image, ground_truths)
    s = "Segmentation"
    numGT = size( ground_truths, 2)
    PR = 0

    #flatten the image
    test_image_flat = Array{UInt16}(vec( test_image ))

    for k in 1:numGT
        PR += randindex(test_image_flat, vec( ground_truths[k][s] ))[2]
    end
    PR /= numGT
    return PR
end
