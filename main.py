import argparse
import textwrap
from enum import IntEnum, unique
import internal


@unique
class ExitCode(IntEnum):
    UNKNOWN_ERROR = 1
    UNKNOWN_ERROR2 = 2


def setup_argparse():
    parser = argparse.ArgumentParser(
        prog='LEGO application',
        description=textwrap.dedent('''\
        python user_app.py\
        python .\\user_app.py --num_clusters 2 --feature --clustering --input ./result/1 --result ./result2 '''),
        epilog=textwrap.dedent(f'''\
        Error codes:
        {ExitCode.UNKNOWN_ERROR} - unspecified error,
        {ExitCode.UNKNOWN_ERROR2} - unspecified error2'''),
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-i', '--input', default="./input", help="Input folder (default: ./input)")
    parser.add_argument('-n', '--net', default="xception", help="Network arhitecture (default: xception)")
    parser.add_argument('-a', '--alg', default="kmeans", help="Clustering alghoritm (default: kmeans)")
    parser.add_argument('-u', '--num_clusters', default="10", help="Number of clusters (default: 10)")
    parser.add_argument('-r', '--result', default="./result", help="Output folder (default: ./result)")

    parser.add_argument('-f', '--feature', action='store_true', help="Feature extraction (default: false)")
    parser.add_argument('-p', '--pca', action='store_true', help='sum the integers (default: false)')
    parser.add_argument('-c', '--clustering', action='store_true', help="Clustering(default: false)")
    parser.add_argument('-t', '--test', action='store_true', help="test (default: false)")
    args = parser.parse_args()

    return args


def main():
    args = setup_argparse()

    print("input            " + args.input)
    print("net              " + args.net)
    print("alg              " + args.alg)
    print("num_clusters     " + args.num_clusters)
    print("result           " + args.result)
    print("feature          " + str(args.feature))
    print("pca              " + str(args.pca))
    print("clustering       " + str(args.clustering))
    print("test             " + str(args.test))

    if args.feature:
        print("Feature extraction")
        internal.img_to_feature_vectors(args.input, args.net, args.test)

    if args.pca:
        print("PCA")
        internal.pca()

    if args.clustering:
        print("Clustering")
        internal.clustering(args.alg, int(args.num_clusters), args.test, args.result)


main()
