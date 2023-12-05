"""
Alternative entry point to work better with DTU Health Tech way of passing arguments.
This follows what is already online for DeepLoc 2.0
"""
import argparse
import DeepLocPro


class FakeArgs():
    def __init__(self, **kwargs):
        # store all kwargs as attributes.
        self.__dict__.update(kwargs)

    def __repr__(self):
        return str(self.__dict__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--FASTA_PATH", type=str, help="Input protein sequences in the FASTA format"
    )
    parser.add_argument(
        "--OUTPUT_PATH", type=str, default="./outputs/", help="Output directory"
    )
    parser.add_argument(
        "--NUM_THREADS",
        default=os.cpu_count(),
        type=int,
        help="Number of threads to use."
    )
    parser.add_argument(
        "--GROUP", 
        default="any",
        choices=['any', 'archaea', 'positive', 'negative'],
        type=str,
        help="Group of input."
    )
    parser.add_argument(
        "--JOBID", type=str, help="Job id from webface"
    )
    parser.add_argument(
        "--FORMAT", default='long', choices=['long','short'],type=str, help="Output format"
    )
    args = parser.parse_args()


    input_args = FakeArgs(
        fasta=args.FASTA_PATH,
        output=args.OUTPUT_PATH,
        device="cpu",
        plot=False if args.FORMAT == 'short' else True,
        group=args.GROUP,
    )

    DeepLocPro.deeplocpro.main(input_args)

