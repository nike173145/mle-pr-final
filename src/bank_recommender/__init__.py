"""Training and inference package for bank-product recommendations."""

from bank_recommender.constants import PRODUCT_COLUMNS
from bank_recommender.model import BankProductRecommender

__all__ = ["BankProductRecommender", "PRODUCT_COLUMNS"]
