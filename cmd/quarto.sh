#! /bin/sh

if [ "$1" = "post-render" ]; then
    rm ./evcxr_pkg/*/.gitignore || exit 0
fi
