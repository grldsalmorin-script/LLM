# Safety Rules

AI recommendations for accounting should not directly post journals or change ledgers without validation. The safer pattern is to return structured recommendations, cite supporting evidence, require human approval when confidence is low or business impact is high, and log the command ID plus source context used to produce the recommendation.
