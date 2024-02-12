const withMT = require("@material-tailwind/html/utils/withMT");
 
module.exports = withMT({
    mode: "all",
    content: [
        // include all rust, html and css files in the src directory
        "./src/**/*.{rs,html,css}",
        // include all html files in the output (dist) directory
        "*.html",
        "doc/*.html",
        "web/**/*.html",
    ],
    theme: {
        extend: {},
    },
    plugins: [],
});
