# Version number should always be updated before pushing commits onto the main branch (or when merging branches onto the main).
# Version number should have the following format:
# A.B.C.<desc>
#     A = Major Release: May break backwards compatibility
#     B = Minor Release: New features but not critical, should be backwards compatible to previous minor releases
#     C = Bug/Hot Fixes: Bug fixes or very minor features and improvements
# 1.2.0.dev1   # Development Build (May be only a partial implementation of a feature / bug-fix)
# 1.2.0.a1     # Alpha Build (All implementation should be done, but not all features for the release are in place; lots of bugs)
# 1.2.0.b1     # Beta Release (All features are in place; minor changes might be needed based on user feedback; some bugs)
# 1.2.0.rc1    # Release Candidate (All features are in place; no changes should be made unless essential; few bugs)
# 1.2.X        # Final Release (X are bug fixes and patches that do not fall on the regular release cycle)

# Note that a particular *version* of TidalPy may be in alpha, beta, dev, etc. that does not necessarily
#   reflect the state of TidalPy as a whole project (which is currently still in Alpha).

version = '0.0.1.dev7'