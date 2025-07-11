for build_id in $(cibuildwheel --print-build-identifiers); do
    echo "Building for $build_id"
    cibuildwheel --output-dir dist --only "$build_id" || echo "Build failed for $build_id"
done
