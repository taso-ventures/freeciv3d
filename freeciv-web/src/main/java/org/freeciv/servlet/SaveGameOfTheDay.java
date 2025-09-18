package org.freeciv.servlet;

import org.apache.commons.io.FileUtils;

import javax.imageio.ImageIO;
import jakarta.servlet.http.HttpServlet;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.stream.Collectors;

/**
 * Saves a game of the day image.
 *
 * URL: /save_game_of_the_day
 */
public class SaveGameOfTheDay extends HttpServlet {

    private static final String MAP_DST_IMG_PATHS = "/var/lib/tomcat10/webapps/data/";

    // Added detailed logging for debugging
    @Override
    public void doPost(HttpServletRequest request, HttpServletResponse response)
            throws IOException {
        System.out.println("Received POST request at /save_game_of_the_day");
        try (BufferedReader reader = request.getReader()) {
            String image = reader.lines()
                    .collect(Collectors.joining(System.lineSeparator()))
                    .replace("data:image/png;base64,", "");
            System.out.println("Extracted Base64 image string");

            byte[] image_of_the_day = Base64.getDecoder().decode(image.getBytes(StandardCharsets.UTF_8));
            System.out.println("Decoded image bytes, size: " + image_of_the_day.length);

            if (image_of_the_day.length > 15000000) {
                System.out.println("Image too big.");
                response.setStatus(HttpServletResponse.SC_BAD_REQUEST);
                response.getWriter().write("Image too big.");
                return;
            }

            try (ByteArrayInputStream bais = new ByteArrayInputStream(image_of_the_day)) {
                BufferedImage bufferedImage = ImageIO.read(bais);
                if (bufferedImage != null) {
                    System.out.println("BufferedImage dimensions: " + bufferedImage.getWidth() + "x" + bufferedImage.getHeight());
                    if (bufferedImage.getWidth() > 100 && bufferedImage.getWidth() < 10000
                            && bufferedImage.getHeight() > 100 && bufferedImage.getHeight() < 10000) {
                        File mapimg = new File(MAP_DST_IMG_PATHS + "game_of_the_day.png");
                        if (!mapimg.getParentFile().exists() && !mapimg.getParentFile().mkdirs()) {
                            System.err.println("Failed to create directory: " + mapimg.getParentFile());
                            response.setStatus(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
                            return;
                        }
                        FileUtils.writeByteArrayToFile(mapimg, image_of_the_day);
                        System.out.println("Image successfully written to: " + mapimg.getAbsolutePath());
                    } else {
                        System.out.println("Invalid image dimensions.");
                        response.setStatus(HttpServletResponse.SC_BAD_REQUEST);
                        response.getWriter().write("Invalid image dimensions.");
                    }
                } else {
                    System.out.println("Failed to create BufferedImage from input stream.");
                    response.setStatus(HttpServletResponse.SC_BAD_REQUEST);
                    response.getWriter().write("Failed to create image.");
                }
            }
        } catch (IOException | IllegalArgumentException ex) {
            System.err.println("Error processing image: " + ex.getMessage());
            response.setStatus(HttpServletResponse.SC_BAD_REQUEST);
            response.getWriter().write("Error processing image: " + ex.getMessage());
        }
    }
}
