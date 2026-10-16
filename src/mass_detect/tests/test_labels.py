"""Unit tests of the label rules of notebook 03: median, mean and mode of the ratings, rounding and the binary scale."""
import numpy as np

from src.mass_detect.scripts.cohort import median_rating, mode_rating, rounded_rating, suspicious

def test_rounding_is_half_up(): # 2.5 becomes 3 and 3.5 becomes 4, unlike numpy, which rounds .5 to the even number
    assert list(rounded_rating([1.5, 2.5, 3.5, 4.49, 2.25])) == [2, 3, 4, 4, 2]

def test_median_of_the_ratings(): # The examples of the notebook: an even number of readers can give a value ending in .5
    assert median_rating([2, 4]) == 3
    assert median_rating([1, 1, 3, 4]) == 2
    assert median_rating([3, 4, 5, 5]) == 4.5 and rounded_rating(median_rating([3, 4, 5, 5])) == 5

def test_mode_and_its_tie_rule(): # The most frequent rating, and the median of the tied ratings on a tie
    assert mode_rating([1, 1, 3, 4]) == 1
    assert mode_rating([3, 4, 5, 5]) == 5
    assert mode_rating([2, 4]) == 3                     # two readers who disagree
    assert mode_rating([2, 2, 4, 4]) == 3
    assert mode_rating([1, 2, 5]) == 2                  # three different ratings: the median of the three

def test_blank_ratings_do_not_count(): # A rating left blank (NaN) is ignored, and a nodule with no rating has no label
    assert median_rating([2, np.nan, 4, 4]) == 4
    assert mode_rating([np.nan, 3]) == 3
    assert np.isnan(median_rating([np.nan])) and np.isnan(mode_rating([]))

def test_one_reader_gives_the_same_label_with_every_rule(): # With a single reader there is nothing to combine
    for rating in range(1, 6):
        assert median_rating([rating]) == mode_rating([rating]) == np.mean([rating]) == rating

def test_binary_scale(): # Suspicious only for ratings 4 and 5
    assert list(suspicious([1, 2, 3, 4, 5])) == [0, 0, 0, 1, 1]
