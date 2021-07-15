import pkg_resources

from myia.infer.infnode import inferrers

for entry_point in pkg_resources.iter_entry_points("myia.plugins"):
    entry_point.load()(inferrers=inferrers)
